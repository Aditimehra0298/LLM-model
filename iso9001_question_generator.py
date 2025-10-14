import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import argparse
import sys

class ISO9001QuestionGenerator:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.data = self.load_data()
        
    def load_data(self):
        """Load and parse the ISO 9001 CSV data"""
        try:
            # Try to read the CSV file with more flexible parsing
            try:
                # First attempt with strict error handling disabled
                df = pd.read_csv(self.csv_file_path, error_bad_lines=False, warn_bad_lines=True)
            except:
                try:
                    # For pandas >= 1.3.0 where error_bad_lines is deprecated
                    df = pd.read_csv(self.csv_file_path, on_bad_lines='skip')
                except:
                    try:
                        # Try with python engine which is more flexible
                        df = pd.read_csv(self.csv_file_path, engine='python')
                    except:
                        # Last resort - try with very flexible settings
                        df = pd.read_csv(self.csv_file_path, sep=',', quoting=3, engine='python')
            
            print(f"CSV loaded successfully. Found {len(df)} rows.")
            
            # Rename columns to match expected field names if needed
            column_mapping = {
                'clause': 'clause',
                'description': 'description',
                'applicability': 'applicability',
                'critical_question': 'critical_question',
                'option_a': 'option_a',
                'option_b': 'option_b',
                'option_c': 'option_c',
                'option_d': 'option_d',
                'learning_objective': 'learning_objective',
                'discussion_object': 'discussion_object',
                'gaps_a': 'gaps_a',
                'gaps_b': 'gaps_b',
                'gaps_c': 'gaps_c',
                'gaps_d': 'gaps_d'
            }
            
            # Apply column mapping where possible
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def get_clauses(self):
        """Get all available clauses from the data"""
        if self.data is not None and 'clause' in self.data.columns:
            return self.data['clause'].dropna().unique().tolist()
        return []
    
    def generate_question(self, clause=None):
        """Generate a question based on a specific clause or random clause"""
        if self.data is None:
            return None
        
        # Filter by clause if specified
        if clause and 'clause' in self.data.columns:
            filtered_data = self.data[self.data['clause'] == clause]
            if filtered_data.empty:
                return None
        else:
            # Get a random row if no clause specified
            filtered_data = self.data
        
        # Select a random row from the filtered data
        row = filtered_data.sample(1).iloc[0]
        
        # Create question dictionary with safe gets
        question = {
            'clause': row.get('clause', '') if hasattr(row, 'get') else row['clause'] if 'clause' in row else '',
            'description': row.get('description', '') if hasattr(row, 'get') else row['description'] if 'description' in row else '',
            'critical_question': row.get('critical_question', '') if hasattr(row, 'get') else row['critical_question'] if 'critical_question' in row else '',
            'options': [
                row.get('option_a', '') if hasattr(row, 'get') else row['option_a'] if 'option_a' in row else '',
                row.get('option_b', '') if hasattr(row, 'get') else row['option_b'] if 'option_b' in row else '',
                row.get('option_c', '') if hasattr(row, 'get') else row['option_c'] if 'option_c' in row else '',
                row.get('option_d', '') if hasattr(row, 'get') else row['option_d'] if 'option_d' in row else ''
            ],
            'learning_objective': row.get('learning_objective', '') if hasattr(row, 'get') else row['learning_objective'] if 'learning_objective' in row else '',
            'discussion_object': row.get('discussion_object', '') if hasattr(row, 'get') else row['discussion_object'] if 'discussion_object' in row else '',
            'gaps': [
                row.get('gaps_a', '') if hasattr(row, 'get') else row['gaps_a'] if 'gaps_a' in row else '',
                row.get('gaps_b', '') if hasattr(row, 'get') else row['gaps_b'] if 'gaps_b' in row else '',
                row.get('gaps_c', '') if hasattr(row, 'get') else row['gaps_c'] if 'gaps_c' in row else '',
                row.get('gaps_d', '') if hasattr(row, 'get') else row['gaps_d'] if 'gaps_d' in row else ''
            ]
        }
        
        return question

class ISO9001QuestionGeneratorApp:
    def __init__(self, root, generator):
        self.root = root
        self.generator = generator
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface"""
        self.root.title("ISO 9001 Question Generator")
        self.root.geometry("800x600")
        
        # Create frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        self.question_frame = ttk.Frame(self.root, padding="10")
        self.question_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control elements
        ttk.Label(self.control_frame, text="Select Clause:").grid(row=0, column=0, padx=5, pady=5)
        
        self.clause_var = tk.StringVar()
        self.clause_dropdown = ttk.Combobox(self.control_frame, textvariable=self.clause_var)
        self.clause_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(self.control_frame, text="Generate Question", command=self.generate_question).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Save Question", command=self.save_question).grid(row=0, column=3, padx=5, pady=5)
        
        # Question display elements
        self.question_text = tk.Text(self.question_frame, wrap=tk.WORD, height=20, width=80)
        self.question_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Load clauses
        self.load_clauses()
        
    def load_clauses(self):
        """Load clauses into the dropdown"""
        clauses = self.generator.get_clauses()
        self.clause_dropdown['values'] = clauses
        if clauses:
            self.clause_dropdown.current(0)
    
    def generate_question(self):
        """Generate and display a question"""
        selected_clause = self.clause_var.get() if self.clause_var.get() else None
        question = self.generator.generate_question(selected_clause)
        
        if question:
            self.display_question(question)
        else:
            messagebox.showerror("Error", "Failed to generate question")
    
    def display_question(self, question):
        """Display the question in the text widget"""
        self.question_text.delete(1.0, tk.END)
        
        self.question_text.insert(tk.END, f"Clause: {question['clause']}\n\n")
        self.question_text.insert(tk.END, f"Description: {question['description']}\n\n")
        self.question_text.insert(tk.END, f"Critical Question: {question['critical_question']}\n\n")
        
        self.question_text.insert(tk.END, "Options:\n")
        for i, option in enumerate(question['options']):
            if option:
                self.question_text.insert(tk.END, f"{chr(65+i)}. {option}\n")
        
        self.question_text.insert(tk.END, f"\nLearning Objective: {question['learning_objective']}\n\n")
        self.question_text.insert(tk.END, f"Discussion Object: {question['discussion_object']}\n\n")
        
        self.question_text.insert(tk.END, "Gaps:\n")
        for i, gap in enumerate(question['gaps']):
            if gap:
                self.question_text.insert(tk.END, f"{chr(65+i)}. {gap}\n")
        
        # Store the current question
        self.current_question = question
    
    def save_question(self):
        """Save the current question to a file"""
        if not hasattr(self, 'current_question'):
            messagebox.showerror("Error", "No question to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_question, f, indent=4)
                messagebox.showinfo("Success", f"Question saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save question: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ISO 9001 Question Generator')
    parser.add_argument('--csv', type=str, help='Path to the CSV file', default='ISO 9001 questions.csv')
    parser.add_argument('--count', type=int, help='Number of questions to generate', default=3)
    parser.add_argument('--clause', type=str, help='Specific clause to generate questions for')
    parser.add_argument('--output', type=str, help='Output file for generated questions')
    parser.add_argument('--gui', action='store_true', help='Launch the GUI')
    
    args = parser.parse_args()
    
    # Create the question generator
    generator = ISO9001QuestionGenerator(args.csv)
    
    # If GUI mode is requested
    if args.gui:
        root = tk.Tk()
        app = ISO9001QuestionGeneratorApp(root, generator)
        root.mainloop()
    else:
        # Generate questions in command line mode
        questions = []
        for _ in range(args.count):
            question = generator.generate_question(args.clause)
            if question:
                questions.append(question)
                print(f"\nQuestion {_ + 1}:")
                print(f"Clause: {question['clause']}")
                print(f"Description: {question['description']}")
                print(f"Critical Question: {question['critical_question']}")
                print("Options:")
                for i, option in enumerate(question['options']):
                    if option:
                        print(f"{chr(65+i)}. {option}")
        
        # Save to output file if specified
        if args.output and questions:
            try:
                with open(args.output, 'w') as f:
                    json.dump(questions, f, indent=4)
                print(f"\nSaved {len(questions)} questions to {args.output}")
            except Exception as e:
                print(f"Error saving to file: {e}")

if __name__ == "__main__":
    main()