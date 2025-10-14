import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class ISO27001QuestionGenerator:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.data = self.load_data()
        
    def load_data(self):
        """Load and parse the ISO 27001 CSV data"""
        try:
            # Skip the first two rows which contain header information
            df = pd.read_csv(self.csv_file_path, skiprows=2)
            
            print(f"CSV loaded successfully. Found {len(df)} rows.")
            print(f"Columns: {df.columns.tolist()}")
            
            # Rename columns to match expected field names
            column_mapping = {
                'clause': 'clause',
                'description': 'description',
                'applicability': 'applicability',
                'critical_question': 'critical_question',
                'format_required': 'format_required',
                'format_no': 'format_no',
                'option_a': 'option_a',
                'option_b': 'option_b',
                'option_c': 'option_c',
                'option_d': 'option_d',
                'learning_objective for workshop': 'learning_objective for workshop',
                'Object for Discussion  for workshop': 'Object for Discussion  for workshop',
                'lEARNING OBJECTIVE FOR SELF PACED COURSE': 'lEARNING OBJECTIVE FOR SELF PACED COURSE',
                'Object for discussion for a Self pacced course': 'Object for discussion for a Self pacced course',
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
            'clause': row.get('clause', '') if 'clause' in row else '',
            'description': row.get('description', '') if 'description' in row else '',
            'critical_question': row.get('critical_question', '') if 'critical_question' in row else '',
            'options': [
                row.get('option_a', '') if 'option_a' in row else '',
                row.get('option_b', '') if 'option_b' in row else '',
                row.get('option_c', '') if 'option_c' in row else '',
                row.get('option_d', '') if 'option_d' in row else ''
            ],
            'learning_objective_workshop': row.get('learning_objective for workshop', '') if 'learning_objective for workshop' in row else '',
            'discussion_object_workshop': row.get('Object for Discussion  for workshop', '') if 'Object for Discussion  for workshop' in row else '',
            'learning_objective_self_paced': row.get('lEARNING OBJECTIVE FOR SELF PACED COURSE', '') if 'lEARNING OBJECTIVE FOR SELF PACED COURSE' in row else '',
            'discussion_object_self_paced': row.get('Object for discussion for a Self pacced course', '') if 'Object for discussion for a Self pacced course' in row else '',
            'gaps': [
                row.get('gaps_a', '') if 'gaps_a' in row else '',
                row.get('gaps_b', '') if 'gaps_b' in row else '',
                row.get('gaps_c', '') if 'gaps_c' in row else '',
                row.get('gaps_d', '') if 'gaps_d' in row else ''
            ]
        }
        
        return question

class ISO27001QuestionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISO 27001 Question Generator")
        self.root.geometry("900x700")
        
        # Default CSV path
        self.csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "ISO 27001 internal audit Tool development - Sheet5.csv")
        
        # Initialize the question generator
        self.question_generator = ISO27001QuestionGenerator(self.csv_path)
        
        # Create the UI
        self.create_widgets()
        
        # Generate a sample question immediately
        self.generate_and_display_question()
        
    def create_widgets(self):
        # Frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # CSV file selection
        ttk.Label(control_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.csv_path_var = tk.StringVar(value=self.question_generator.csv_file_path)
        ttk.Entry(control_frame, textvariable=self.csv_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_csv).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Load CSV", command=self.load_csv).grid(row=0, column=3, padx=5, pady=5)
        
        # Clause selection
        ttk.Label(control_frame, text="Clause:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.clause_var = tk.StringVar()
        self.clause_dropdown = ttk.Combobox(control_frame, textvariable=self.clause_var, width=20)
        self.clause_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Generate button
        ttk.Button(control_frame, text="Generate Question", command=self.generate_question).grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Frame for question display
        question_frame = ttk.LabelFrame(self.root, text="Generated Question", padding="10")
        question_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Question display
        ttk.Label(question_frame, text="Clause:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.clause_display = ttk.Label(question_frame, text="")
        self.clause_display.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(question_frame, text="Description:").grid(row=1, column=0, sticky=tk.NW, padx=5, pady=5)
        self.description_display = tk.Text(question_frame, wrap=tk.WORD, width=70, height=3)
        self.description_display.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.description_display.config(state=tk.DISABLED)
        
        ttk.Label(question_frame, text="Critical Question:").grid(row=2, column=0, sticky=tk.NW, padx=5, pady=5)
        self.question_display = tk.Text(question_frame, wrap=tk.WORD, width=70, height=2)
        self.question_display.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        self.question_display.config(state=tk.DISABLED)
        
        # Options
        ttk.Label(question_frame, text="Options:").grid(row=3, column=0, sticky=tk.NW, padx=5, pady=5)
        self.options_frame = ttk.Frame(question_frame)
        self.options_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.option_displays = []
        option_labels = ["A", "B", "C", "D"]
        for i in range(4):
            ttk.Label(self.options_frame, text=f"{option_labels[i]}:").grid(row=i, column=0, sticky=tk.NW, padx=5, pady=2)
            option_text = tk.Text(self.options_frame, wrap=tk.WORD, width=65, height=2)
            option_text.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            option_text.config(state=tk.DISABLED)
            self.option_displays.append(option_text)
        
        # Learning objectives
        ttk.Label(question_frame, text="Learning Objectives:").grid(row=4, column=0, sticky=tk.NW, padx=5, pady=5)
        self.learning_frame = ttk.Frame(question_frame)
        self.learning_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.learning_frame, text="Workshop:").grid(row=0, column=0, sticky=tk.NW, padx=5, pady=2)
        self.workshop_lo_display = tk.Text(self.learning_frame, wrap=tk.WORD, width=65, height=2)
        self.workshop_lo_display.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.workshop_lo_display.config(state=tk.DISABLED)
        
        ttk.Label(self.learning_frame, text="Self-paced:").grid(row=1, column=0, sticky=tk.NW, padx=5, pady=2)
        self.self_paced_lo_display = tk.Text(self.learning_frame, wrap=tk.WORD, width=65, height=2)
        self.self_paced_lo_display.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.self_paced_lo_display.config(state=tk.DISABLED)
        
        # Discussion objects
        ttk.Label(question_frame, text="Discussion Objects:").grid(row=5, column=0, sticky=tk.NW, padx=5, pady=5)
        self.discussion_frame = ttk.Frame(question_frame)
        self.discussion_frame.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.discussion_frame, text="Workshop:").grid(row=0, column=0, sticky=tk.NW, padx=5, pady=2)
        self.workshop_disc_display = tk.Text(self.discussion_frame, wrap=tk.WORD, width=65, height=2)
        self.workshop_disc_display.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.workshop_disc_display.config(state=tk.DISABLED)
        
        ttk.Label(self.discussion_frame, text="Self-paced:").grid(row=1, column=0, sticky=tk.NW, padx=5, pady=2)
        self.self_paced_disc_display = tk.Text(self.discussion_frame, wrap=tk.WORD, width=65, height=2)
        self.self_paced_disc_display.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.self_paced_disc_display.config(state=tk.DISABLED)
        
        # Gaps
        ttk.Label(question_frame, text="Gaps:").grid(row=6, column=0, sticky=tk.NW, padx=5, pady=5)
        self.gaps_frame = ttk.Frame(question_frame)
        self.gaps_frame.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.gaps_displays = []
        gap_labels = ["A", "B", "C", "D"]
        for i in range(4):
            ttk.Label(self.gaps_frame, text=f"{gap_labels[i]}:").grid(row=i, column=0, sticky=tk.NW, padx=5, pady=2)
            gap_text = tk.Text(self.gaps_frame, wrap=tk.WORD, width=65, height=1)
            gap_text.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            gap_text.config(state=tk.DISABLED)
            self.gaps_displays.append(gap_text)
    
    def browse_csv(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select ISO 27001 CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.csv_path_var.set(file_path)
    
    def load_csv(self):
        """Load the selected CSV file"""
        csv_path = self.csv_path_var.get()
        if os.path.exists(csv_path):
            self.question_generator = ISO27001QuestionGenerator(csv_path)
            self.load_clauses()
            messagebox.showinfo("Success", "CSV file loaded successfully!")
        else:
            messagebox.showerror("Error", "CSV file not found!")
    
    def load_clauses(self):
        """Load clauses into the dropdown"""
        clauses = self.question_generator.get_clauses()
        if clauses:
            self.clause_dropdown['values'] = ["Random"] + clauses
            self.clause_dropdown.current(0)
        else:
            self.clause_dropdown['values'] = ["No clauses found"]
            self.clause_dropdown.current(0)
    
    def set_text(self, text_widget, content):
        """Set text in a text widget"""
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def generate_question(self):
        """Generate and display a question"""
        clause = self.clause_var.get()
        if clause == "Random" or clause == "No clauses found":
            clause = None
        
        question = self.question_generator.generate_question(clause)
        if question:
            # Display question details
            self.clause_display.config(text=f"{question['clause']}")
            self.set_text(self.description_display, question['description'])
            self.set_text(self.question_display, question['critical_question'])
            
            # Display options
            for i, option_text in enumerate(question['options']):
                self.set_text(self.option_displays[i], option_text)
            
            # Display learning objectives
            self.set_text(self.workshop_lo_display, question['learning_objective_workshop'])
            self.set_text(self.self_paced_lo_display, question['learning_objective_self_paced'])
            
            # Display discussion objects
            self.set_text(self.workshop_disc_display, question['discussion_object_workshop'])
            self.set_text(self.self_paced_disc_display, question['discussion_object_self_paced'])
            
            # Display gaps
            for i, gap_text in enumerate(question['gaps']):
                self.set_text(self.gaps_displays[i], gap_text)
        else:
            messagebox.showerror("Error", "Failed to generate question!")

def generate_and_display_question(self):
        """Generate and display a question"""
        question = self.question_generator.generate_question()
        if question:
            # Display question details in console for quick testing
            print("\n===== GENERATED QUESTION =====")
            print(f"Clause: {question['clause']}")
            print(f"Description: {question['description']}")
            print(f"Critical Question: {question['critical_question']}")
            print("\nOptions:")
            for i, option in enumerate(question['options']):
                print(f"  {chr(65+i)}. {option}")
            print("\nLearning Objective (Workshop):", question['learning_objective_workshop'])
            print("Discussion Object (Workshop):", question['discussion_object_workshop'])
            print("\nLearning Objective (Self-paced):", question['learning_objective_self_paced'])
            print("Discussion Object (Self-paced):", question['discussion_object_self_paced'])
            print("\nGaps:")
            for i, gap in enumerate(question['gaps']):
                print(f"  {chr(65+i)}. {gap}")
            print("==============================\n")
            
            # Also update the UI if it exists
            try:
                # Display question details
                self.clause_display.config(text=f"{question['clause']}")
                self.set_text(self.description_display, question['description'])
                self.set_text(self.question_display, question['critical_question'])
                
                # Display options
                for i, option_text in enumerate(question['options']):
                    self.set_text(self.option_displays[i], option_text)
                
                # Display learning objectives
                self.set_text(self.workshop_lo_display, question['learning_objective_workshop'])
                self.set_text(self.self_paced_lo_display, question['learning_objective_self_paced'])
                
                # Display discussion objects
                self.set_text(self.workshop_disc_display, question['discussion_object_workshop'])
                self.set_text(self.self_paced_disc_display, question['discussion_object_self_paced'])
                
                # Display gaps
                for i, gap_text in enumerate(question['gaps']):
                    self.set_text(self.gaps_displays[i], gap_text)
            except:
                # UI elements might not be fully initialized yet
                pass
        else:
            print("Failed to generate question!")

def main():
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='ISO 27001 Question Generator')
    parser.add_argument('--csv', type=str, help='Path to the CSV file', 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "ISO 27001 internal audit Tool development - Sheet5.csv"))
    parser.add_argument('--count', type=int, help='Number of questions to generate', default=3)
    parser.add_argument('--clause', type=str, help='Specific clause to generate questions for', default=None)
    parser.add_argument('--output', type=str, help='Output file to save questions', default=None)
    parser.add_argument('--gui', action='store_true', help='Launch the GUI interface')
    
    args = parser.parse_args()
    
    # Initialize the generator
    print(f"Loading CSV from: {args.csv}")
    generator = ISO27001QuestionGenerator(args.csv)
    
    if args.gui:
        # Launch GUI if requested
        root = tk.Tk()
        app = ISO27001QuestionGeneratorApp(root)
        root.mainloop()
        return
    
    # Generate questions
    print(f"\nGenerating {args.count} questions" + (f" for clause {args.clause}" if args.clause else ""))
    
    questions = []
    for i in range(args.count):
        question = generator.generate_question(args.clause)
        if question:
            print(f"\n--- Question {i+1} ---")
            print(f"Clause: {question['clause']}")
            print(f"Description: {question['description'][:100]}..." if len(question['description']) > 100 else question['description'])
            print(f"Critical Question: {question['critical_question']}")
            print("Options:")
            for j, opt in enumerate(question['options']):
                if opt:
                    print(f"  {chr(65+j)}. {opt[:100]}..." if len(opt) > 100 else opt)
            
            # Add to questions list for potential output
            questions.append(question)
        else:
            print(f"Failed to generate question {i+1}")
    
    # Save to output file if specified
    if args.output and questions:
        try:
            import json
            with open(args.output, 'w') as f:
                json.dump(questions, f, indent=2)
            print(f"\nSuccessfully saved {len(questions)} questions to {args.output}")
        except Exception as e:
            print(f"Error saving to output file: {e}")
    
    # Only start GUI if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        root = tk.Tk()
        app = ISO27001QuestionGeneratorApp(root)
        root.mainloop()
    
if __name__ == "__main__":
    import sys
    main()

if __name__ == "__main__":
    main()