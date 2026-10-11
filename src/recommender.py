"""
recommender.py - Cluster-aware hybrid course recommendation engine and learning path generator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ClusterAwareRecommender:
    """
    Hybrid Course Recommendation Engine combining:
    1. Content-based cosine similarity on course metadata.
    2. Cluster-aware collaborative popularity and peer affinity.
    3. Rating-weighted quality scoring.
    4. Pedagogical level progression alignment.
    5. Step-by-step personalized learning paths (Foundation -> Core -> Mastery).
    """

    def __init__(
        self,
        courses_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        learners_df: pd.DataFrame,
        w_content: float = 0.35,
        w_cluster: float = 0.35,
        w_rating: float = 0.15,
        w_progression: float = 0.15
    ):
        self.courses_df = courses_df.copy()
        self.transactions_df = transactions_df.copy()
        self.learners_df = learners_df.copy()
        
        # Recommendation weights
        self.w_content = w_content
        self.w_cluster = w_cluster
        self.w_rating = w_rating
        self.w_progression = w_progression

        self.course_similarity_matrix: np.ndarray = None
        self.course_features_df: pd.DataFrame = None
        self.cluster_course_popularity: Dict[int, Dict[str, float]] = {}

        self._build_course_representations()
        self._compute_cluster_popularities()

    def _build_course_representations(self):
        """
        Builds content feature vectors for all 60 courses using text attributes and numerical features.
        """
        courses = self.courses_df.copy()
        
        # Create rich course descriptor string
        courses['content_text'] = (
            courses['CourseName'] + " " +
            courses['CourseCategory'] + " " +
            courses['CourseLevel'] + " " +
            courses['CourseType'] + " " +
            "Category_" + courses['CourseCategory'].str.replace(' ', '_') + " " +
            "Level_" + courses['CourseLevel']
        )

        # TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(courses['content_text']).toarray()

        # Normalized numerical features
        scaler = MinMaxScaler()
        num_features = scaler.fit_transform(courses[['CourseRating', 'CourseDuration', 'CoursePrice']])

        # Combine TF-IDF text features and normalized numeric features
        combined_features = np.hstack([tfidf_matrix * 0.7, num_features * 0.3])
        
        # Precompute item-to-item cosine similarity matrix
        self.course_similarity_matrix = cosine_similarity(combined_features)
        self.course_features_df = courses
        self.course_id_to_idx = {cid: idx for idx, cid in enumerate(courses['CourseID'])}
        self.idx_to_course_id = {idx: cid for idx, cid in enumerate(courses['CourseID'])}

    def _compute_cluster_popularities(self):
        """
        Computes normalized course enrollment counts and rating weights within each learner cluster.
        """
        if 'cluster' not in self.learners_df.columns:
            return

        tx_merged = self.transactions_df.merge(
            self.learners_df[['UserID', 'cluster']], on='UserID', how='inner'
        )
        tx_merged = tx_merged.merge(
            self.courses_df[['CourseID', 'CourseRating']], on='CourseID', how='inner'
        )

        for cluster_id in self.learners_df['cluster'].unique():
            cluster_tx = tx_merged[tx_merged['cluster'] == cluster_id]
            if cluster_tx.empty:
                continue

            # Popularity = enrollments * average course rating
            counts = cluster_tx.groupby('CourseID').size()
            ratings = cluster_tx.groupby('CourseID')['CourseRating'].mean()
            score = counts * (ratings / 5.0)

            # Max-normalize
            if score.max() > 0:
                normalized_score = (score / score.max()).to_dict()
            else:
                normalized_score = {}

            self.cluster_course_popularity[cluster_id] = normalized_score

    def get_recommendations_for_learner(
        self,
        user_id: str,
        top_n: int = 5,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None,
        filter_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generates top-N personalized course recommendations for a given learner.
        """
        learner_row = self.learners_df[self.learners_df['UserID'] == user_id]
        
        if learner_row.empty:
            # Fallback to cold-start top rated / popular courses
            return self._get_cold_start_recommendations(top_n, filter_category, filter_level, filter_type)

        learner = learner_row.iloc[0]
        user_cluster = int(learner.get('cluster', 0))
        learning_depth = float(learner.get('learning_depth_index', 0.5))
        primary_category = str(learner.get('primary_category', ''))
        preferred_level = str(learner.get('primary_course_level', 'Beginner'))

        # Get courses already enrolled by user
        enrolled_course_ids = set(
            self.transactions_df[self.transactions_df['UserID'] == user_id]['CourseID'].unique()
        )

        recommendation_records = []
        cluster_pop_dict = self.cluster_course_popularity.get(user_cluster, {})

        for idx, course in self.courses_df.iterrows():
            cid = course['CourseID']
            if cid in enrolled_course_ids:
                continue  # Skip already enrolled courses

            # 1. Content-based similarity score
            if enrolled_course_ids:
                enrolled_indices = [self.course_id_to_idx[c] for c in enrolled_course_ids if c in self.course_id_to_idx]
                if enrolled_indices:
                    content_sim = float(np.mean(self.course_similarity_matrix[idx, enrolled_indices]))
                else:
                    content_sim = 0.5
            else:
                content_sim = 0.5

            # Boost content similarity if matching preferred category
            if course['CourseCategory'] == primary_category:
                content_sim = min(1.0, content_sim + 0.25)

            # 2. Cluster affinity score
            cluster_score = cluster_pop_dict.get(cid, 0.2)

            # 3. Normalized course rating
            rating_norm = float(course['CourseRating']) / 5.0

            # 4. Level progression fit
            level_fit = self._calculate_level_progression_fit(
                user_depth=learning_depth,
                user_level=preferred_level,
                course_level=course['CourseLevel']
            )

            # Hybrid composite score
            hybrid_score = (
                self.w_content * content_sim +
                self.w_cluster * cluster_score +
                self.w_rating * rating_norm +
                self.w_progression * level_fit
            )

            # Explanation tags
            reasons = []
            if course['CourseCategory'] == primary_category:
                reasons.append(f"Matches your primary focus ({primary_category})")
            if cluster_score > 0.6:
                reasons.append(f"Trending among peers in your segment")
            if course['CourseRating'] >= 4.5:
                reasons.append(f"Top-rated curriculum ({course['CourseRating']} / 5.0)")
            if level_fit >= 0.8:
                reasons.append(f"Ideal progression step for your level")
            if not reasons:
                reasons.append("High overall relevance & satisfaction score")

            recommendation_records.append({
                'CourseID': cid,
                'CourseName': course['CourseName'],
                'CourseCategory': course['CourseCategory'],
                'CourseLevel': course['CourseLevel'],
                'CourseType': course['CourseType'],
                'CoursePrice': course['CoursePrice'],
                'CourseDuration': course['CourseDuration'],
                'CourseRating': course['CourseRating'],
                'RecommendationScore': round(float(hybrid_score) * 100, 1),
                'ContentSimilarity': round(float(content_sim) * 100, 1),
                'ClusterAffinity': round(float(cluster_score) * 100, 1),
                'ProgressionFit': round(float(level_fit) * 100, 1),
                'Reasoning': " • ".join(reasons)
            })

        recs_df = pd.DataFrame(recommendation_records)
        if recs_df.empty:
            return pd.DataFrame()

        # Apply user filters if specified
        if filter_category and filter_category != "All":
            recs_df = recs_df[recs_df['CourseCategory'] == filter_category]
        if filter_level and filter_level != "All":
            recs_df = recs_df[recs_df['CourseLevel'] == filter_level]
        if filter_type and filter_type != "All":
            recs_df = recs_df[recs_df['CourseType'] == filter_type]

        recs_df = recs_df.sort_values(by='RecommendationScore', ascending=False).head(top_n)
        return recs_df.reset_index(drop=True)

    def _calculate_level_progression_fit(self, user_depth: float, user_level: str, course_level: str) -> float:
        """
        Scores pedagogical transition appropriateness based on learning depth index.
        """
        # Level hierarchy
        level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        c_lvl = level_map.get(course_level, 1)

        if user_depth < 0.45:  # Beginner orientation
            if c_lvl == 1: return 1.0
            elif c_lvl == 2: return 0.7
            else: return 0.3
        elif user_depth < 0.70:  # Intermediate progression
            if c_lvl == 2: return 1.0
            elif c_lvl == 3: return 0.8
            else: return 0.5
        else:  # Mastery / Advanced orientation
            if c_lvl == 3: return 1.0
            elif c_lvl == 2: return 0.7
            else: return 0.4

    def generate_personalized_learning_path(self, user_id: str, target_category: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Constructs a tailored 3-step sequential roadmap:
        Step 1: Foundation (Beginner)
        Step 2: Core Practical (Intermediate)
        Step 3: Advanced Mastery / Capstone (Advanced)
        """
        learner_row = self.learners_df[self.learners_df['UserID'] == user_id]
        if not learner_row.empty:
            primary_cat = target_category or learner_row.iloc[0].get('primary_category', 'Data Science')
        else:
            primary_cat = target_category or 'Data Science'

        cat_courses = self.courses_df[self.courses_df['CourseCategory'] == primary_cat]
        if cat_courses.empty:
            cat_courses = self.courses_df

        # Group by level and pick top rated
        beg_c = cat_courses[cat_courses['CourseLevel'] == 'Beginner'].sort_values('CourseRating', ascending=False)
        int_c = cat_courses[cat_courses['CourseLevel'] == 'Intermediate'].sort_values('CourseRating', ascending=False)
        adv_c = cat_courses[cat_courses['CourseLevel'] == 'Advanced'].sort_values('CourseRating', ascending=False)

        pathway = []
        if not beg_c.empty:
            c = beg_c.iloc[0]
            pathway.append({
                'Stage': 'Stage 1: Foundational Core',
                'CourseID': c['CourseID'],
                'CourseName': c['CourseName'],
                'Level': c['CourseLevel'],
                'Duration': f"{c['CourseDuration']} Hours",
                'Rating': c['CourseRating'],
                'Price': "Free" if c['CoursePrice'] == 0 else f"${c['CoursePrice']:.2f}",
                'Objective': f"Master fundamentals of {primary_cat} with essential conceptual tools."
            })

        if not int_c.empty:
            c = int_c.iloc[0]
            pathway.append({
                'Stage': 'Stage 2: Applied Competency',
                'CourseID': c['CourseID'],
                'CourseName': c['CourseName'],
                'Level': c['CourseLevel'],
                'Duration': f"{c['CourseDuration']} Hours",
                'Rating': c['CourseRating'],
                'Price': "Free" if c['CoursePrice'] == 0 else f"${c['CoursePrice']:.2f}",
                'Objective': f"Develop intermediate applied problem-solving and industry workflows."
            })

        if not adv_c.empty:
            c = adv_c.iloc[0]
            pathway.append({
                'Stage': 'Stage 3: Advanced Mastery',
                'CourseID': c['CourseID'],
                'CourseName': c['CourseName'],
                'Level': c['CourseLevel'],
                'Duration': f"{c['CourseDuration']} Hours",
                'Rating': c['CourseRating'],
                'Price': "Free" if c['CoursePrice'] == 0 else f"${c['CoursePrice']:.2f}",
                'Objective': f"Achieve enterprise-grade specialization, optimization, and capstone deployment."
            })

        return pathway

    def _get_cold_start_recommendations(
        self, top_n: int, filter_category: Optional[str], filter_level: Optional[str], filter_type: Optional[str]
    ) -> pd.DataFrame:
        """
        Fallback high-quality course list for uninitiated users.
        """
        df = self.courses_df.copy()
        if filter_category and filter_category != "All":
            df = df[df['CourseCategory'] == filter_category]
        if filter_level and filter_level != "All":
            df = df[df['CourseLevel'] == filter_level]
        if filter_type and filter_type != "All":
            df = df[df['CourseType'] == filter_type]

        df['RecommendationScore'] = (df['CourseRating'] / 5.0 * 100).round(1)
        df['Reasoning'] = "High platform rating & universal popularity for incoming learners"
        return df.sort_values(by='CourseRating', ascending=False).head(top_n)
