import pandas as pd
import sys

def validate_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check number of columns
        if len(df.columns) != 10:
            print(f"❌ Error: File has {len(df.columns)} columns, but expected 10 columns")
            return False
            
        # Check column names
        expected_columns = [
            'heart_rate',
            'blood_pressure',
            'oxygen',
            'temperature',
            'respiratory_rate',
            'blood_glucose',
            'cholesterol',
            'bmi',
            'age',
            'activity_level'
        ]
        
        if not all(col in df.columns for col in expected_columns):
            print("❌ Error: Missing required columns. Expected columns are:")
            for col in expected_columns:
                print(f"  - {col}")
            return False
            
        # Check for missing values
        if df.isna().any().any():
            print("❌ Error: File contains missing values")
            return False
            
        # Check if all values are numeric
        if not all(df[col].dtype in ['int64', 'float64'] for col in expected_columns):
            print("❌ Error: All columns must contain numeric values")
            return False
            
        # Check value ranges
        ranges = {
            'heart_rate': (40, 200),
            'blood_pressure': (60, 200),
            'oxygen': (70, 100),
            'temperature': (95, 105),
            'respiratory_rate': (12, 30),
            'blood_glucose': (70, 200),
            'cholesterol': (100, 300),
            'bmi': (15, 50),
            'age': (0, 120),
            'activity_level': (1, 5)  # 1: Sedentary, 5: Very Active
        }
        
        for col, (min_val, max_val) in ranges.items():
            if (df[col] < min_val).any() or (df[col] > max_val).any():
                print(f"❌ Error: {col} values should be between {min_val} and {max_val}")
                return False
        
        print("✅ File validation successful!")
        print("\nFirst few rows of your data:")
        print(df.head())
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_data.py <path_to_csv_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    validate_csv(file_path) 