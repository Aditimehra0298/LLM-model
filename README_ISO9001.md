# ISO 9001 Question Generator

A tool for generating audit questions based on ISO 9001 standard clauses.

## Features

- Generate random questions from ISO 9001 clauses
- Filter questions by specific clauses
- Save generated questions to JSON format
- Optional GUI interface for interactive use

## Usage

### Command Line Interface

```bash
# Generate 3 random questions
python iso9001_question_generator.py --count 3

# Generate 5 questions and save to a file
python iso9001_question_generator.py --count 5 --output questions.json

# Generate questions for a specific clause
python iso9001_question_generator.py --clause 4.1 --count 2

# Launch the GUI interface
python iso9001_question_generator.py --gui
```

### Command Line Arguments

- `--csv`: Path to the CSV file (default: "ISO 9001 questions.csv")
- `--count`: Number of questions to generate (default: 3)
- `--clause`: Specific clause to generate questions for
- `--output`: Output file for generated questions
- `--gui`: Launch the GUI interface

## CSV Format

The generator expects a CSV file with the following columns:
- clause
- description
- applicability
- critical_question
- option_a, option_b, option_c, option_d
- learning_objective
- discussion_object
- gaps_a, gaps_b, gaps_c, gaps_d

## Example Output

```json
[
  {
    "clause": "4.1",
    "description": "Context of the organization",
    "critical_question": "How does the organization determine external and internal issues?",
    "options": [
      "The organization conducts regular SWOT analysis",
      "The organization relies on customer feedback only",
      "The organization addresses issues only when critical",
      "The organization has no formal process"
    ],
    "learning_objective": "Understand how to identify and monitor issues",
    "discussion_object": "Discuss methods for determining context",
    "gaps": [
      "No systematic approach to identifying issues",
      "Limited scope of analysis",
      "Reactive approach to issues",
      "Lack of formal process"
    ]
  }
]
```