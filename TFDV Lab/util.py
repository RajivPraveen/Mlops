import pandas as pd


def add_extra_rows(df):
    rows = [
        {
            'PassengerId': 9999,
            'Survived': 1,
            'Pclass': 5,
            'Name': 'Doe, Mr. John',
            'Sex': 'unknown',
            'Age': -5,
            'SibSp': 0,
            'Parch': 0,
            'Ticket': 'XXXXX',
            'Fare': 999.99,
            'Cabin': 'Z99',
            'Embarked': 'X'
        },
        {
            'PassengerId': 9998,
            'Survived': 0,
            'Pclass': 1,
            'Name': 'Smith, Mrs. Jane',
            'Sex': 'female',
            'Age': 200,
            'SibSp': 0,
            'Parch': 0,
            'Ticket': 'PC 99999',
            'Fare': 50.0,
            'Embarked': 'S'
        },
        {
            'PassengerId': 9997,
            'Survived': 1,
            'Pclass': 2,
            'Name': 'Lee, Mr. Bruce',
            'Age': 35,
            'SibSp': 10,
            'Parch': 0,
            'Ticket': 'A/5 00000',
            'Fare': -20.0,
            'Cabin': 'B99',
            'Embarked': 'Q'
        },
        {
            'PassengerId': 9996,
            'Survived': 0,
            'Pclass': 3,
            'Name': 'Garcia, Ms. Maria',
            'Sex': 'non-binary',
            'Age': 0,
            'SibSp': 0,
            'Parch': 15,
            'Ticket': '00000',
            'Fare': 7.75,
            'Cabin': None,
            'Embarked': 'S'
        }
    ]

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    return df
