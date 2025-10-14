# FSSC 22000 Question Generator

A tool for generating assessment questions based on the FSSC 22000 Food Safety System Certification standard.

## Features

- Generate random questions from a comprehensive database of FSSC 22000 clauses
- Filter questions by specific clauses
- Specify the number of questions to generate
- Output questions in JSON format
- GUI interface for easy use

## Usage

### Command Line Interface

```bash
python fssc22000_question_generator.py --csv "FSSC 22000 questions.csv" --count 5 --output fssc22000_questions.json
```

### GUI Interface

```bash
python fssc22000_question_generator.py --gui
```

### Arguments

- `--csv`: Path to the CSV file containing FSSC 22000 questions (required)
- `--count`: Number of questions to generate (default: 5)
- `--clauses`: Filter questions by specific clauses (optional)
- `--output`: Output JSON file path (optional)
- `--gui`: Launch the GUI interface (optional)

## CSV Format

The CSV file should contain the following columns:

- `clause`: The FSSC 22000 clause number (e.g., "4.1", "5.2")
- `description`: Brief description of the clause
- `applicability`: Who the clause applies to
- `critical_question`: The main question to ask
- `option_a`, `option_b`, `option_c`, `option_d`: Multiple choice options
- `gaps_a`, `gaps_b`, `gaps_c`, `gaps_d`: Corresponding gaps for each option
- `learning_objective`: What should be learned from this question
- `discussion_object`: Topics for discussion

## Example Output

```json
[
  {
    "clause": "7.2",
    "description": "Competence",
    "applicability": "All organizations",
    "critical_question": "How should personnel competence be managed?",
    "options": [
      {
        "text": "The organization shall determine necessary competence and ensure personnel are competent",
        "gap": "Comprehensive competence management"
      },
      {
        "text": "Competence is determined only for technical positions",
        "gap": "Limited scope of competence management"
      },
      {
        "text": "Training is provided only for regulatory compliance",
        "gap": "Compliance-driven training only"
      },
      {
        "text": "Training is provided only when problems arise",
        "gap": "Reactive approach to training"
      }
    ],
    "learning_objective": "Understand competence requirements",
    "discussion_objective": "Discuss effective competence management"
  }
]
```

## Question Database

The included CSV file contains 30+ comprehensive questions covering all major clauses of the FSSC 22000 standard:

- Core ISO 22000 requirements (clauses 4.1-10.3)
- FSSC 22000 additional requirements:
  - Food Defense
  - Food Fraud Prevention
  - Allergen Management
  - Environmental Monitoring
  - Transport and Delivery

Each question includes:
- The specific clause reference
- A critical question about implementation
- Four multiple-choice options representing different levels of implementation
- Learning and discussion objectives
- Potential gaps for each implementation level

## Requirements

- Python 3.6+
- pandas
- tkinter (for GUI interface)

## Installation

No special installation is required beyond the Python dependencies:

```bash
pip install pandas
```

Tkinter is included with most Python installations.