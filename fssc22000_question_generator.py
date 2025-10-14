#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd

class FSSC22000QuestionGenerator:
    """
    A class to generate questions based on FSSC 22000 standard clauses.
    """
    
    def __init__(self, csv_path=None):
        """
        Initialize the FSSC 22000 Question Generator.
        
        Args:
            csv_path (str, optional): Path to the CSV file containing FSSC 22000 questions.
        """
        self.csv_path = csv_path
        self.data = []
        self.column_mapping = {
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
        
        if csv_path:
            self.load_data(csv_path)
    
    def load_data(self, csv_path):
        """
        Load data from the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file.
            
        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        try:
            # Try multiple approaches to handle potential CSV parsing issues
            try:
                # First attempt: Standard CSV reading
                self.data = pd.read_csv(csv_path).to_dict('records')
                return True
            except Exception as e:
                try:
                    # Second attempt: Handle quoted fields
                    self.data = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL).to_dict('records')
                    return True
                except Exception as e:
                    try:
                        # Third attempt: More flexible quoting
                        self.data = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL).to_dict('records')
                        return True
                    except Exception as e:
                        try:
                            # Fourth attempt: Use Python engine for messy data
                            self.data = pd.read_csv(csv_path, engine='python', on_bad_lines='skip').to_dict('records')
                            return True
                        except Exception as e:
                            # Final attempt: Most flexible approach
                            self.data = pd.read_csv(csv_path, engine='python', sep=',', 
                                                   error_bad_lines=False, 
                                                   warn_bad_lines=True).to_dict('records')
                            return True
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return False
    
    def get_clauses(self):
        """
        Get a list of all FSSC 22000 clauses from the loaded data.
        
        Returns:
            list: List of unique clauses.
        """
        if not self.data:
            return []
        
        clauses = []
        for item in self.data:
            clause = item.get(self.column_mapping['clause'])
            if clause and clause not in clauses:
                clauses.append(clause)
        
        return sorted(clauses)
    
    def generate_questions(self, count=1, clauses=None):
        """
        Generate random questions based on FSSC 22000 clauses.
        
        Args:
            count (int, optional): Number of questions to generate. Defaults to 1.
            clauses (list, optional): List of clauses to filter by. Defaults to None.
            
        Returns:
            list: List of generated questions.
        """
        if not self.data:
            return []
        
        filtered_data = self.data
        if clauses:
            filtered_data = [item for item in self.data if item.get(self.column_mapping['clause']) in clauses]
        
        if not filtered_data:
            return []
        
        # Ensure we don't try to generate more questions than available
        count = min(count, len(filtered_data))
        
        # Randomly select questions
        selected_items = random.sample(filtered_data, count)
        
        questions = []
        for item in selected_items:
            question = {
                'clause': item.get(self.column_mapping['clause'], ''),
                'description': item.get(self.column_mapping['description'], ''),
                'applicability': item.get(self.column_mapping['applicability'], ''),
                'critical_question': item.get(self.column_mapping['critical_question'], ''),
                'options': [
                    {'text': item.get(self.column_mapping['option_a'], ''), 'gap': item.get(self.column_mapping['gaps_a'], '')},
                    {'text': item.get(self.column_mapping['option_b'], ''), 'gap': item.get(self.column_mapping['gaps_b'], '')},
                    {'text': item.get(self.column_mapping['option_c'], ''), 'gap': item.get(self.column_mapping['gaps_c'], '')},
                    {'text': item.get(self.column_mapping['option_d'], ''), 'gap': item.get(self.column_mapping['gaps_d'], '')}
                ],
                'learning_objective': item.get(self.column_mapping['learning_objective'], ''),
                'discussion_objective': item.get(self.column_mapping['discussion_object'], '')
            }
            questions.append(question)
        
        return questions


class FSSC22000QuestionGeneratorApp:
    """
    GUI application for the FSSC 22000 Question Generator.
    """
    
    def __init__(self, root, generator=None):
        """
        Initialize the GUI application.
        
        Args:
            root (tk.Tk): Root Tkinter window.
            generator (FSSC22000QuestionGenerator, optional): Question generator instance.
        """
        self.root = root
        self.root.title("FSSC 22000 Question Generator")
        self.root.geometry("800x600")
        
        self.generator = generator or FSSC22000QuestionGenerator()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Frame for CSV selection
        csv_frame = ttk.LabelFrame(self.root, text="CSV File")
        csv_frame.pack(fill="x", padx=10, pady=5)
        
        self.csv_path_var = tk.StringVar()
        ttk.Entry(csv_frame, textvariable=self.csv_path_var, width=50).pack(side="left", padx=5, pady=5)
        ttk.Button(csv_frame, text="Browse...", command=self.browse_csv).pack(side="left", padx=5, pady=5)
        ttk.Button(csv_frame, text="Load", command=self.load_csv).pack(side="left", padx=5, pady=5)
        
        # Frame for question generation options
        options_frame = ttk.LabelFrame(self.root, text="Generation Options")
        options_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(options_frame, text="Number of Questions:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.count_var = tk.IntVar(value=5)
        ttk.Spinbox(options_frame, from_=1, to=100, textvariable=self.count_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Clause selection
        ttk.Label(options_frame, text="Clauses:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.clauses_frame = ttk.Frame(options_frame)
        self.clauses_frame.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Output options
        ttk.Label(options_frame, text="Output File:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.output_path_var = tk.StringVar()
        ttk.Entry(options_frame, textvariable=self.output_path_var, width=40).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(options_frame, text="Browse...", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5, sticky="w")
        
        # Generate button
        ttk.Button(options_frame, text="Generate Questions", command=self.generate_questions).grid(row=3, column=1, padx=5, pady=10)
        
        # Results area
        results_frame = ttk.LabelFrame(self.root, text="Generated Questions")
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap="word", width=80, height=20)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=scrollbar.set)
    
    def browse_csv(self):
        """Open file dialog to select CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select FSSC 22000 Questions CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.csv_path_var.set(filepath)
    
    def browse_output(self):
        """Open file dialog to select output file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Generated Questions",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.output_path_var.set(filepath)
    
    def load_csv(self):
        """Load CSV data and update clause checkboxes."""
        csv_path = self.csv_path_var.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        
        if not os.path.exists(csv_path):
            messagebox.showerror("Error", f"File not found: {csv_path}")
            return
        
        self.generator = FSSC22000QuestionGenerator(csv_path)
        
        # Clear existing clause checkboxes
        for widget in self.clauses_frame.winfo_children():
            widget.destroy()
        
        # Create clause checkboxes
        clauses = self.generator.get_clauses()
        self.clause_vars = {}
        
        for i, clause in enumerate(clauses):
            var = tk.BooleanVar(value=True)
            self.clause_vars[clause] = var
            ttk.Checkbutton(self.clauses_frame, text=clause, variable=var).grid(row=i//3, column=i%3, sticky="w", padx=5)
        
        messagebox.showinfo("Success", f"Loaded {len(clauses)} clauses from CSV.")
    
    def generate_questions(self):
        """Generate questions based on selected options."""
        if not self.generator.data:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return
        
        count = self.count_var.get()
        
        # Get selected clauses
        selected_clauses = [clause for clause, var in self.clause_vars.items() if var.get()]
        
        if not selected_clauses:
            messagebox.showerror("Error", "Please select at least one clause.")
            return
        
        # Generate questions
        questions = self.generator.generate_questions(count, selected_clauses)
        
        if not questions:
            messagebox.showerror("Error", "No questions could be generated with the selected criteria.")
            return
        
        # Display questions
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, json.dumps(questions, indent=2))
        
        # Save to file if specified
        output_path = self.output_path_var.get()
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, indent=2)
                messagebox.showinfo("Success", f"Generated {len(questions)} questions and saved to {output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save output: {str(e)}")
        else:
            messagebox.showinfo("Success", f"Generated {len(questions)} questions.")


def main():
    """Main function to parse arguments and run the generator."""
    parser = argparse.ArgumentParser(description="Generate FSSC 22000 questions")
    parser.add_argument("--csv", help="Path to the CSV file with FSSC 22000 questions")
    parser.add_argument("--count", type=int, default=5, help="Number of questions to generate")
    parser.add_argument("--clauses", nargs="+", help="Filter questions by specific clauses")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    
    args = parser.parse_args()
    
    if args.gui:
        root = tk.Tk()
        app = FSSC22000QuestionGeneratorApp(root)
        if args.csv:
            app.csv_path_var.set(args.csv)
            app.load_csv()
        root.mainloop()
    else:
        if not args.csv:
            print("Error: CSV file path is required.")
            parser.print_help()
            sys.exit(1)
        
        generator = FSSC22000QuestionGenerator(args.csv)
        questions = generator.generate_questions(args.count, args.clauses)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=2)
            print(f"Generated {len(questions)} questions and saved to {args.output}")
        else:
            print(json.dumps(questions, indent=2))


if __name__ == "__main__":
    main()