"""
data_loader.py - Data ingestion, schema validation, and preprocessing for EduPro platform.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class EduProDataLoader:
    """
    Handles ingestion, validation, and joining of EduPro platform dataset sheets.
    """

    def __init__(self, excel_path: str = "EduPro Online Platform.xlsx"):
        self.excel_path = excel_path
        self.users_df: pd.DataFrame = pd.DataFrame()
        self.courses_df: pd.DataFrame = pd.DataFrame()
        self.transactions_df: pd.DataFrame = pd.DataFrame()
        self.teachers_df: pd.DataFrame = pd.DataFrame()
        self.merged_df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads all sheets from the Excel file and performs basic cleaning.
        """
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel dataset not found at {self.excel_path}")

        excel_file = pd.ExcelFile(self.excel_path)
        required_sheets = ['Users', 'Courses', 'Transactions', 'Teachers']
        for sheet in required_sheets:
            if sheet not in excel_file.sheet_names:
                raise ValueError(f"Missing required sheet '{sheet}' in {self.excel_path}")

        self.users_df = pd.read_excel(self.excel_path, sheet_name='Users')
        self.courses_df = pd.read_excel(self.excel_path, sheet_name='Courses')
        self.transactions_df = pd.read_excel(self.excel_path, sheet_name='Transactions')
        self.teachers_df = pd.read_excel(self.excel_path, sheet_name='Teachers')

        # Clean strings and strip whitespace
        for df in [self.users_df, self.courses_df, self.transactions_df, self.teachers_df]:
            for col in df.select_dtypes(include=['object', 'string']).columns:
                df[col] = df[col].astype(str).str.strip()

        # Parse dates
        if 'TransactionDate' in self.transactions_df.columns:
            self.transactions_df['TransactionDate'] = pd.to_datetime(
                self.transactions_df['TransactionDate'], errors='coerce'
            )

        # Ensure numeric types
        self.users_df['Age'] = pd.to_numeric(self.users_df['Age'], errors='coerce').fillna(25).astype(int)
        self.courses_df['CoursePrice'] = pd.to_numeric(self.courses_df['CoursePrice'], errors='coerce').fillna(0.0)
        self.courses_df['CourseDuration'] = pd.to_numeric(self.courses_df['CourseDuration'], errors='coerce').fillna(10.0)
        self.courses_df['CourseRating'] = pd.to_numeric(self.courses_df['CourseRating'], errors='coerce').fillna(4.0)
        self.transactions_df['Amount'] = pd.to_numeric(self.transactions_df['Amount'], errors='coerce').fillna(0.0)

        # Create master merged transaction view
        self._create_merged_view()

        return {
            'users': self.users_df,
            'courses': self.courses_df,
            'transactions': self.transactions_df,
            'teachers': self.teachers_df,
            'merged': self.merged_df
        }

    def _create_merged_view(self):
        """
        Merges transactions with courses, users, and teacher metadata.
        """
        tx = self.transactions_df.copy()
        tx = tx.merge(self.users_df, on='UserID', how='left')
        tx = tx.merge(self.courses_df, on='CourseID', how='left', suffixes=('', '_course'))
        tx = tx.merge(self.teachers_df, on='TeacherID', how='left', suffixes=('', '_teacher'))
        self.merged_df = tx

    def get_summary_stats(self) -> Dict[str, any]:
        """
        Returns key summary statistics of the raw data.
        """
        return {
            'total_users': len(self.users_df),
            'total_courses': len(self.courses_df),
            'total_transactions': len(self.transactions_df),
            'total_teachers': len(self.teachers_df),
            'date_range': (
                self.transactions_df['TransactionDate'].min().strftime('%Y-%m-%d')
                if not self.transactions_df.empty else 'N/A',
                self.transactions_df['TransactionDate'].max().strftime('%Y-%m-%d')
                if not self.transactions_df.empty else 'N/A'
            ),
            'categories': self.courses_df['CourseCategory'].unique().tolist(),
            'total_revenue': float(self.transactions_df['Amount'].sum())
        }
