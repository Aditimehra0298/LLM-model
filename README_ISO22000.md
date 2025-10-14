# ISO 22000:2018 Question Generator

A tool for generating audit questions based on the ISO 22000:2018 Food Safety Management System standard.

## Features

- Generate random audit questions from ISO 22000:2018 clauses
- Filter questions by specific clauses
- Specify the number of questions to generate
- Output questions in JSON format
- Simple command-line interface
- Optional GUI interface for easier use

## Usage

### Command Line Interface

```bash
python iso22000_question_generator.py --csv "ISO 22000 questions.csv" --count 5 --output "iso22000_questions.json"
```

### GUI Interface

```bash
python iso22000_question_generator.py --gui
```

## Arguments

- `--csv`: Path to the CSV file containing ISO 22000:2018 questions (default: "ISO 22000 questions.csv")
- `--count`: Number of questions to generate (default: 5)
- `--clauses`: Specific clauses to filter questions by (e.g., "4.1 7.2 8.5")
- `--output`: Output JSON file path (default: "iso22000_questions.json")
- `--gui`: Launch the graphical user interface

## CSV Format

The CSV file should contain the following columns:
- Clause: The ISO 22000:2018 clause number (e.g., "4.1")
- Description: Brief description of the clause
- Applicability: Who the clause applies to
- Critical Question: The main audit question
- Options: Multiple-choice options (separated by "|")
- Learning Objective: Educational goal of the question
- Discussion Objective: Topics for further discussion
- Potential Gaps: Potential non-conformities (separated by "|")

## Example Output

```json
[
  {
    "clause": "8.7",
    "description": "Control of monitoring and measuring",
    "applicability": "All organizations",
    "critical_question": "How should monitoring and measuring equipment be controlled?",
    "options": [
      {
        "text": "The organization shall provide evidence that monitoring and measuring equipment is fit for purpose",
        "gap": "Comprehensive calibration system"
      },
      {
        "text": "Equipment is calibrated only for critical measurements",
        "gap": "Limited scope of calibration"
      },
      {
        "text": "Equipment is calibrated only during audit preparation",
        "gap": "Audit-driven calibration"
      },
      {
        "text": "The organization has minimal calibration controls",
        "gap": "Minimal calibration controls"
      }
    ],
    "learning_objective": "Understand equipment control requirements",
    "discussion_objective": "Discuss effective calibration management"
  }
]
```

## Question Database

The included `ISO 22000 questions.csv` file contains over 30 questions covering all major clauses of ISO 22000:2018, including:

- Context of the organization (Clause 4)
- Leadership (Clause 5)
- Planning (Clause 6)
- Support (Clause 7)
- Operation (Clause 8)
- Performance evaluation (Clause 9)
- Improvement (Clause 10)

## Requirements

- Python 3.6 or higher
- pandas
- tkinter (for GUI interface)

## Installation

No special installation required beyond the Python dependencies:

```bash
pip install pandas
```

Tkinter is included with most Python installations for the GUI interface.