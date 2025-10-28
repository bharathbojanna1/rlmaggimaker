import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import xgboost as xgb

# ===================== NEURAL NETWORKS (same as training) =====================
class Actor(nn.Module):
    """Policy network (actor) for SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


# ===================== RECIPE OPTIMIZER WITH FIXED INGREDIENT =====================
class RecipeOptimizer:
    """Test trained RL agent by fixing one ingredient and optimizing others"""
    
    def __init__(self, model_path, xgboost_model, dataset_path='maggi_dataset.txt'):
        """
        Load trained RL agent and setup optimizer
        
        Args:
            model_path: Path to trained SAC agent (.pth file)
            xgboost_model: Your trained XGBoost model
            dataset_path: Path to maggi_dataset.txt
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xgb_model = xgboost_model
        
        # Load dataset
        self.df = pd.read_csv(dataset_path)
        self.action_bounds = self._calculate_bounds()
        self.ingredient_names = list(self.action_bounds.keys())
        self.ideal_ratios = self._calculate_ideal_ratios()
        
        # Load trained actor
        state_dim = 9  # 1 (taste) + 8 (ingredients)
        action_dim = 8
        self.actor = Actor(state_dim, action_dim).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()
        
        print(f"✓ Loaded trained RL agent from: {model_path}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Available ingredients: {self.ingredient_names}\n")
    
    def _calculate_bounds(self):
        """Extract bounds from dataset"""
        bounds = {}
        columns = ['Maggi_Packets', 'Masala_Sachets', 'Water_ml', 'Onions_g',
                   'ChilliPowder_tbsp', 'Turmeric_tbsp', 'Salt_tbsp', 'CookingTime_min']
        
        for col in columns:
            bounds[col] = (float(self.df[col].min()), float(self.df[col].max()))
        
        return bounds
    
    def _calculate_ideal_ratios(self):
        """Calculate ideal ratios from high-quality recipes"""
        high_quality = self.df[self.df['TasteScore'] >= 95]
        
        ratios = {}
        ratios['masala_per_packet'] = (high_quality['Masala_Sachets'] / high_quality['Maggi_Packets']).mean()
        ratios['water_per_packet'] = (high_quality['Water_ml'] / high_quality['Maggi_Packets']).mean()
        ratios['onions_per_packet'] = (high_quality['Onions_g'] / high_quality['Maggi_Packets']).mean()
        ratios['chilli_per_packet'] = (high_quality['ChilliPowder_tbsp'] / high_quality['Maggi_Packets']).mean()
        ratios['turmeric_per_packet'] = (high_quality['Turmeric_tbsp'] / high_quality['Maggi_Packets']).mean()
        ratios['salt_per_packet'] = (high_quality['Salt_tbsp'] / high_quality['Maggi_Packets']).mean()
        ratios['time_per_packet'] = (high_quality['CookingTime_min'] / high_quality['Maggi_Packets']).mean()
        
        return ratios
    
    def _create_state(self, fixed_ingredient, fixed_value):
        """Create state with one ingredient fixed"""
        min_val, max_val = self.action_bounds[fixed_ingredient]
        normalized_fixed = (fixed_value - min_val) / (max_val - min_val)
        
        previous_taste = 0.85  # Target high quality
        baseline = np.array([0.5] * len(self.ingredient_names))
        
        fixed_idx = self.ingredient_names.index(fixed_ingredient)
        baseline[fixed_idx] = normalized_fixed
        
        state = np.concatenate([[previous_taste], baseline])
        return state.astype(np.float32)
    
    def _denormalize_action(self, action, fixed_ingredient, fixed_value):
        """Convert actor output to recipe while respecting fixed ingredient"""
        ingredients = []
        fixed_idx = self.ingredient_names.index(fixed_ingredient)
        
        # Infer packets from fixed ingredient
        if fixed_ingredient == 'Maggi_Packets':
            packets = fixed_value
        elif fixed_ingredient == 'Masala_Sachets':
            packets = fixed_value / self.ideal_ratios['masala_per_packet']
        elif fixed_ingredient == 'Water_ml':
            packets = fixed_value / self.ideal_ratios['water_per_packet']
        elif fixed_ingredient == 'Onions_g':
            packets = fixed_value / self.ideal_ratios['onions_per_packet']
        elif fixed_ingredient == 'ChilliPowder_tbsp':
            packets = fixed_value / self.ideal_ratios['chilli_per_packet']
        elif fixed_ingredient == 'Turmeric_tbsp':
            packets = fixed_value / self.ideal_ratios['turmeric_per_packet']
        elif fixed_ingredient == 'Salt_tbsp':
            packets = fixed_value / self.ideal_ratios['salt_per_packet']
        elif fixed_ingredient == 'CookingTime_min':
            packets = fixed_value / self.ideal_ratios['time_per_packet']
        
        packets = np.clip(packets, 
                         self.action_bounds['Maggi_Packets'][0], 
                         self.action_bounds['Maggi_Packets'][1])
        
        # Calculate all ingredients
        for i, name in enumerate(self.ingredient_names):
            if name == fixed_ingredient:
                ingredients.append(fixed_value)
            elif name == 'Maggi_Packets':
                ingredients.append(packets)
            else:
                # Use actor's action to adjust from ideal ratio
                if name == 'Masala_Sachets':
                    ideal = packets * self.ideal_ratios['masala_per_packet']
                    deviation = action[i] * 0.3 * ideal
                elif name == 'Water_ml':
                    ideal = packets * self.ideal_ratios['water_per_packet']
                    deviation = action[i] * 0.15 * ideal
                elif name == 'Onions_g':
                    ideal = packets * self.ideal_ratios['onions_per_packet']
                    deviation = action[i] * 0.3 * ideal
                elif name == 'ChilliPowder_tbsp':
                    ideal = packets * self.ideal_ratios['chilli_per_packet']
                    deviation = action[i] * 0.3 * ideal
                elif name == 'Turmeric_tbsp':
                    ideal = packets * self.ideal_ratios['turmeric_per_packet']
                    deviation = action[i] * 0.3 * ideal
                elif name == 'Salt_tbsp':
                    ideal = packets * self.ideal_ratios['salt_per_packet']
                    deviation = action[i] * 0.3 * ideal
                elif name == 'CookingTime_min':
                    ideal = packets * self.ideal_ratios['time_per_packet']
                    deviation = action[i] * 0.25 * ideal
                
                value = np.clip(ideal + deviation, 
                              self.action_bounds[name][0], 
                              self.action_bounds[name][1])
                ingredients.append(value)
        
        return np.array(ingredients)
    
    def optimize_with_fixed_ingredient(self, fixed_ingredient, fixed_value, num_samples=20):
        """
        Fix one ingredient and optimize all others
        
        Args:
            fixed_ingredient: Name of ingredient to fix (e.g., 'Salt_tbsp')
            fixed_value: Value for the fixed ingredient (e.g., 1.5)
            num_samples: Number of samples to try (default: 20)
        
        Returns:
            best_recipe: Dict with all ingredient quantities
            best_taste: Predicted taste score
        """
        # Validate
        if fixed_ingredient not in self.ingredient_names:
            raise ValueError(f"Invalid ingredient. Choose from: {self.ingredient_names}")
        
        min_val, max_val = self.action_bounds[fixed_ingredient]
        if not (min_val <= fixed_value <= max_val):
            raise ValueError(f"{fixed_ingredient} must be between {min_val} and {max_val}")
        
        print(f"\n{'='*70}")
        print(f"OPTIMIZING RECIPE WITH: {fixed_ingredient} = {fixed_value}")
        print(f"{'='*70}\n")
        
        best_taste = 0
        best_recipe = None
        all_results = []
        
        for i in range(num_samples):
            # Create state
            state = self._create_state(fixed_ingredient, fixed_value)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from trained actor
            with torch.no_grad():
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean).cpu().numpy()[0]
            
            # Convert to recipe
            ingredients = self._denormalize_action(action, fixed_ingredient, fixed_value)
            
            # Predict taste
            ingredients_reshaped = ingredients.reshape(1, -1)
            feature_names = ['Maggi_Packets', 'Masala_Sachets', 'Water_ml', 'Onions_g',
                           'ChilliPowder_tbsp', 'Turmeric_tbsp', 'Salt_tbsp', 'CookingTime_min']
            dmatrix = xgb.DMatrix(ingredients_reshaped, feature_names=feature_names)
            taste = self.xgb_model.predict(dmatrix)[0]
            taste = np.clip(taste, 0, 100)
            
            recipe = {name: float(val) for name, val in zip(self.ingredient_names, ingredients)}
            all_results.append((recipe, taste))
            
            if taste > best_taste:
                best_taste = taste
                best_recipe = recipe
        
        # Display results
        print(f"Tested {num_samples} variations\n")
        print(f"{'='*70}")
        print(f"BEST RECIPE FOUND - Taste Score: {best_taste:.2f}/100")
        print(f"{'='*70}\n")
        
        for ingredient, amount in best_recipe.items():
            marker = " ⬅ FIXED" if ingredient == fixed_ingredient else ""
            print(f"  {ingredient:20s}: {amount:8.2f}{marker}")
        
        # Show ratios
        packets = best_recipe['Maggi_Packets']
        print(f"\n{'─'*70}")
        print(f"PER-PACKET RATIOS:")
        print(f"{'─'*70}")
        print(f"  Masala per packet:      {best_recipe['Masala_Sachets']/packets:.2f}")
        print(f"  Water per packet:       {best_recipe['Water_ml']/packets:.2f} ml")
        print(f"  Onions per packet:      {best_recipe['Onions_g']/packets:.2f} g")
        print(f"  Chilli per packet:      {best_recipe['ChilliPowder_tbsp']/packets:.4f} tbsp")
        print(f"  Turmeric per packet:    {best_recipe['Turmeric_tbsp']/packets:.4f} tbsp")
        print(f"  Salt per packet:        {best_recipe['Salt_tbsp']/packets:.4f} tbsp")
        print(f"  Time per packet:        {best_recipe['CookingTime_min']/packets:.2f} min")
        print()
        
        # Show top 5 results
        all_results.sort(key=lambda x: x[1], reverse=True)
        print(f"{'─'*70}")
        print(f"TOP 5 VARIATIONS:")
        print(f"{'─'*70}")
        for idx, (recipe, taste) in enumerate(all_results[:5]):
            print(f"\n#{idx+1} - Taste: {taste:.2f}")
            print(f"  Packets: {recipe['Maggi_Packets']:.1f}, "
                  f"Water: {recipe['Water_ml']:.0f}ml, "
                  f"Masala: {recipe['Masala_Sachets']:.1f}")
        
        return best_recipe, best_taste
    
    def interactive_test(self):
        """Interactive command-line interface"""
        print("\n" + "="*70)
        print("INTERACTIVE RL AGENT TESTER")
        print("="*70)
        print("\nFix ONE ingredient and the RL agent will optimize ALL others!\n")
        print("Available ingredients:")
        for i, name in enumerate(self.ingredient_names):
            min_val, max_val = self.action_bounds[name]
            print(f"  {i+1}. {name:20s} (valid range: {min_val:.1f} - {max_val:.1f})")
        
        while True:
            print("\n" + "-"*70)
            choice = input("\nEnter ingredient number to fix (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Exiting. Goodbye!")
                break
            
            try:
                idx = int(choice) - 1
                if not (0 <= idx < len(self.ingredient_names)):
                    print("❌ Invalid choice. Try again.")
                    continue
                
                ingredient = self.ingredient_names[idx]
                min_val, max_val = self.action_bounds[ingredient]
                
                value_str = input(f"Enter value for {ingredient} ({min_val:.1f}-{max_val:.1f}): ").strip()
                value = float(value_str)
                
                if not (min_val <= value <= max_val):
                    print(f"❌ Value must be between {min_val:.1f} and {max_val:.1f}")
                    continue
                
                # Optimize recipe
                recipe, taste = self.optimize_with_fixed_ingredient(ingredient, value)
                
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except Exception as e:
                print(f"❌ Error: {e}")


# ===================== MAIN - LOAD AND TEST =====================
if __name__ == "__main__":
    print("="*70)
    print("TRAINED RL AGENT TESTER")
    print("="*70)
    print("\nLoad your trained RL agent and XGBoost model, then test!\n")
    print("Example usage:")
    print("="*70)

import xgboost as xgb

# Load your XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('maggi_xgboost_model.json')

# Load trained RL agent and create optimizer
optimizer = RecipeOptimizer(
    model_path='sac_maggi_agent_constrained.pth',  # Your trained RL agent
    xgboost_model=xgb_model,
    dataset_path='maggi_dataset.txt'
)

# Test 1: Fix Salt to 1.5 tbsp, get optimal other ingredients
recipe, taste = optimizer.optimize_with_fixed_ingredient('Salt_tbsp', 1.5)

# Test 2: Fix Water to 500ml
recipe, taste = optimizer.optimize_with_fixed_ingredient('Water_ml', 500)

# Test 3: Fix Maggi Packets to 5
recipe, taste = optimizer.optimize_with_fixed_ingredient('Maggi_Packets', 5)

# Interactive mode - test multiple scenarios easily
optimizer.interactive_test()

print("="*70)