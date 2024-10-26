
import json

# Sample SQL dictionary and column names for testing
test_cases = [
    {
        "sql_dict": {
            "select": [0, 1],
            "conds": {
                "column_index": [2, 3],
                "operator_index": [0, 1],
                "condition": ["yes", "session"]
            }
        },
        "column_names": ["ID", "Format", "Notes", "Current series"],
        "expected": 'SELECT Format WHERE Notes = "yes" AND Current series != "session"'
    },
    {
        "sql_dict": {
            "select": [0, 0],
            "conds": {
                "column_index": [1],
                "operator_index": [0],
                "condition": ["plugin"]
            }
        },
        "column_names": ["ID", "Format", "Notes", "Current series"],
        "expected": 'SELECT ID WHERE Format = "plugin"'
    },
    # Add more test cases as needed
]

def sql_dict_to_string(sql_dict, column_names):
    """Translate a SQL dictionary into a SQL-like string using column names."""
    try:
        select_clause = f'SELECT {column_names[sql_dict["select"][1]]}' if sql_dict["select"][1] < len(column_names) else "SELECT *"
    except KeyError:
        select_clause = "SELECT *"

    where_clause = ""
    if "conds" in sql_dict and len(sql_dict['conds']) > 0:
        conditions = [
            f'{column_names[col]} {"=" if op == 0 else "!="} "{cond}"'
            for col, op, cond in zip(
                sql_dict['conds']['column_index'],
                sql_dict['conds']['operator_index'],
                sql_dict['conds']['condition']
            ) if col < len(column_names)
        ]
        where_clause = " AND ".join(conditions)

    query = select_clause
    if where_clause:
        query += f' WHERE {where_clause}'
    return query

def run_tests():
    for i, test in enumerate(test_cases, 1):
        result = sql_dict_to_string(test["sql_dict"], test["column_names"])
        print(f"Test case {i}:")
        print(f"Expected: {test['expected']}")
        print(f"Got:      {result}")
        print("PASS" if result == test["expected"] else "FAIL", "\n")

if __name__ == "__main__":
    run_tests()
