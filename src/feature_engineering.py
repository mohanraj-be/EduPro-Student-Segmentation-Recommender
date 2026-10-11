"""
feature_engineering.py - Learner-level feature extraction and aggregation for EduPro.
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from typing import Dict, List, Tuple


class LearnerFeatureEngineer:
    """
    Transforms raw transactional, course, and demographic records into rich,
    learner-centric multidimensional feature profiles for clustering and recommendations.
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.users_df = data_dict['users'].copy()
        self.courses_df = data_dict['courses'].copy()
        self.transactions_df = data_dict['transactions'].copy()
        self.merged_df = data_dict['merged'].copy()
        self.all_categories = sorted(self.courses_df['CourseCategory'].dropna().unique().tolist())
        self.all_levels = ['Beginner', 'Intermediate', 'Advanced']
        self.all_types = ['Free', 'Paid']

    def construct_learner_features(self) -> pd.DataFrame:
        """
        Builds the unified learner profile DataFrame indexed by UserID.
        """
        # Base user profile
        learner_df = self.users_df[['UserID', 'UserName', 'Age', 'Gender', 'Email']].copy()

        # Aggregate transactions per user
        tx_agg = self._compute_transaction_aggregates()
        cat_agg = self._compute_category_preferences()
        level_agg = self._compute_level_and_depth_features()
        behavioral_agg = self._compute_behavioral_and_diversity_features()

        # Merge all components
        features_df = learner_df.merge(tx_agg, on='UserID', how='left')
        features_df = features_df.merge(cat_agg, on='UserID', how='left')
        features_df = features_df.merge(level_agg, on='UserID', how='left')
        features_df = features_df.merge(behavioral_agg, on='UserID', how='left')

        # Fill missing values for users with 0 or sparse transactions
        features_df = self._impute_missing_values(features_df)

        return features_df

    def _compute_transaction_aggregates(self) -> pd.DataFrame:
        """
        Computes core engagement and transaction metrics per learner.
        """
        merged = self.merged_df

        # Basic counts
        agg = merged.groupby('UserID').agg(
            total_transactions=('TransactionID', 'count'),
            total_courses_enrolled=('CourseID', 'nunique'),
            total_spend=('Amount', 'sum'),
            avg_spend_per_transaction=('Amount', 'mean'),
            avg_course_rating_enrolled=('CourseRating', 'mean'),
            total_learning_hours=('CourseDuration', 'sum'),
            avg_course_duration=('CourseDuration', 'mean'),
            first_transaction_date=('TransactionDate', 'min'),
            last_transaction_date=('TransactionDate', 'max')
        ).reset_index()

        # Engagement frequency / active timespan in days
        timespan_days = (agg['last_transaction_date'] - agg['first_transaction_date']).dt.total_seconds() / (24 * 3600)
        timespan_days = timespan_days.clip(lower=1.0)  # at least 1 day
        agg['active_timespan_days'] = timespan_days
        
        # Enrollment frequency: transactions per month (approx 30 days)
        agg['enrollment_frequency_monthly'] = (agg['total_transactions'] / (agg['active_timespan_days'] / 30.0)).clip(upper=30.0)
        
        # Average spend per course
        agg['avg_spend_per_course'] = agg['total_spend'] / agg['total_courses_enrolled'].clip(lower=1)

        return agg

    def _compute_category_preferences(self) -> pd.DataFrame:
        """
        Computes category-specific preferences, category ratios, and primary category.
        """
        merged = self.merged_df

        # Pivot table for category counts per user
        cat_pivot = pd.crosstab(merged['UserID'], merged['CourseCategory'])
        
        # Ensure all catalog categories are present in columns
        for cat in self.all_categories:
            if cat not in cat_pivot.columns:
                cat_pivot[cat] = 0

        # Calculate category distribution ratios
        cat_totals = cat_pivot.sum(axis=1).replace(0, 1)
        cat_ratios = cat_pivot.div(cat_totals, axis=0)
        cat_ratios.columns = [f"cat_ratio_{c.replace(' ', '_')}" for c in cat_ratios.columns]

        # Primary preferred category & category counts
        primary_category = cat_pivot.idxmax(axis=1)
        categories_explored = (cat_pivot > 0).sum(axis=1)
        avg_courses_per_touched_category = cat_totals / categories_explored.replace(0, 1)

        cat_pref_df = pd.DataFrame({
            'UserID': cat_pivot.index,
            'primary_category': primary_category.values,
            'categories_explored_count': categories_explored.values,
            'avg_courses_per_category': avg_courses_per_touched_category.values
        })

        # Join ratios
        cat_pref_df = cat_pref_df.merge(cat_ratios.reset_index(), on='UserID', how='left')
        return cat_pref_df

    def _compute_level_and_depth_features(self) -> pd.DataFrame:
        """
        Computes course level distribution, learning depth index, and preferred level.
        """
        merged = self.merged_df

        level_pivot = pd.crosstab(merged['UserID'], merged['CourseLevel'])
        for lvl in self.all_levels:
            if lvl not in level_pivot.columns:
                level_pivot[lvl] = 0

        level_totals = level_pivot.sum(axis=1).replace(0, 1)
        level_ratios = level_pivot.div(level_totals, axis=0)

        # Learning depth index: (Beginner*1 + Intermediate*2 + Advanced*3) / (Total*3)
        # Scaled between 0.33 (all beginner) and 1.0 (all advanced)
        weighted_score = (
            level_pivot['Beginner'] * 1 +
            level_pivot['Intermediate'] * 2 +
            level_pivot['Advanced'] * 3
        ) / (level_totals * 3.0)

        # Advanced-to-Beginner ratio
        adv_to_beg_ratio = (level_pivot['Advanced'] + 1) / (level_pivot['Beginner'] + 1)

        level_df = pd.DataFrame({
            'UserID': level_pivot.index,
            'primary_course_level': level_pivot.idxmax(axis=1).values,
            'level_ratio_Beginner': level_ratios['Beginner'].values,
            'level_ratio_Intermediate': level_ratios['Intermediate'].values,
            'level_ratio_Advanced': level_ratios['Advanced'].values,
            'learning_depth_index': weighted_score.values,
            'adv_to_beg_ratio': adv_to_beg_ratio.values
        })

        return level_df

    def _compute_behavioral_and_diversity_features(self) -> pd.DataFrame:
        """
        Computes behavioral diversity scores, Shannon entropy, specialization ratio,
        and free-vs-paid orientation.
        """
        merged = self.merged_df

        # Type pivot (Free vs Paid)
        type_pivot = pd.crosstab(merged['UserID'], merged['CourseType'])
        for tp in self.all_types:
            if tp not in type_pivot.columns:
                type_pivot[tp] = 0

        type_totals = type_pivot.sum(axis=1).replace(0, 1)
        free_ratio = type_pivot['Free'] / type_totals
        paid_ratio = type_pivot['Paid'] / type_totals
        preferred_type = type_pivot.idxmax(axis=1)

        # Category entropy per user
        cat_pivot = pd.crosstab(merged['UserID'], merged['CourseCategory'])
        cat_probs = cat_pivot.div(cat_pivot.sum(axis=1).replace(0, 1), axis=0).values
        cat_entropy = [entropy(p) if np.sum(p) > 0 else 0.0 for p in cat_probs]

        # Diversity score = explored categories / total catalog categories (12)
        total_catalog_cats = max(len(self.all_categories), 1)
        diversity_score = (cat_pivot > 0).sum(axis=1) / total_catalog_cats

        # Specialization ratio = max category enrollment / total enrollments
        specialization_ratio = cat_pivot.max(axis=1) / cat_pivot.sum(axis=1).replace(0, 1)

        behavior_df = pd.DataFrame({
            'UserID': type_pivot.index,
            'preferred_course_type': preferred_type.values,
            'free_course_ratio': free_ratio.values,
            'paid_course_ratio': paid_ratio.values,
            'diversity_score': diversity_score.values,
            'category_entropy': cat_entropy,
            'specialization_ratio': specialization_ratio.values
        })

        return behavior_df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values for learners with defaults to ensure clean clustering vectors.
        """
        defaults = {
            'total_transactions': 0,
            'total_courses_enrolled': 0,
            'total_spend': 0.0,
            'avg_spend_per_transaction': 0.0,
            'avg_spend_per_course': 0.0,
            'avg_course_rating_enrolled': 3.5,
            'total_learning_hours': 0.0,
            'avg_course_duration': 10.0,
            'active_timespan_days': 1.0,
            'enrollment_frequency_monthly': 0.0,
            'primary_category': 'Unspecified',
            'categories_explored_count': 0,
            'avg_courses_per_category': 0.0,
            'primary_course_level': 'Beginner',
            'level_ratio_Beginner': 1.0,
            'level_ratio_Intermediate': 0.0,
            'level_ratio_Advanced': 0.0,
            'learning_depth_index': 0.33,
            'adv_to_beg_ratio': 1.0,
            'preferred_course_type': 'Free',
            'free_course_ratio': 1.0,
            'paid_course_ratio': 0.0,
            'diversity_score': 0.0,
            'category_entropy': 0.0,
            'specialization_ratio': 1.0
        }

        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # Fill any remaining category ratios with 0.0
        cat_ratio_cols = [c for c in df.columns if c.startswith('cat_ratio_')]
        for c in cat_ratio_cols:
            df[c] = df[c].fillna(0.0)

        return df

    def get_feature_matrix_for_clustering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extracts and selects numerical feature columns suited for distance-based clustering.
        """
        cluster_feature_cols = [
            'Age',
            'total_courses_enrolled',
            'total_spend',
            'avg_spend_per_course',
            'avg_course_rating_enrolled',
            'total_learning_hours',
            'avg_course_duration',
            'enrollment_frequency_monthly',
            'categories_explored_count',
            'avg_courses_per_category',
            'learning_depth_index',
            'level_ratio_Beginner',
            'level_ratio_Intermediate',
            'level_ratio_Advanced',
            'diversity_score',
            'category_entropy',
            'specialization_ratio',
            'paid_course_ratio'
        ]

        # Filter to existing columns
        selected_cols = [c for c in cluster_feature_cols if c in df.columns]
        feature_matrix = df[selected_cols].copy()

        return feature_matrix, selected_cols
