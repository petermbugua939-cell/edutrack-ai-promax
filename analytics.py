import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from database import SessionLocal
import models
from ai_engine import AdvancedAIEngine

class AdvancedAnalytics:
    def __init__(self):
        self.db = SessionLocal()
        self.ai_engine = AdvancedAIEngine()
        
    def analyze_school_performance(self, school_id: int) -> Dict[str, Any]:
        """Comprehensive school performance analysis"""
        print(f"📊 Analyzing school performance for school {school_id}")
        
        # Get all students in school
        students = self.db.query(models.Student).filter(
            models.Student.school_id == school_id,
            models.Student.is_active == True
        ).all()
        
        if not students:
            return {"error": "No active students found"}
        
        student_ids = [s.id for s in students]
        
        # Get all exam results
        results = self.db.query(models.ExamResult).filter(
            models.ExamResult.student_id.in_(student_ids)
        ).all()
        
        # Get all attendance records (last 90 days)
        ninety_days_ago = datetime.utcnow().date() - timedelta(days=90)
        attendance = self.db.query(models.Attendance).filter(
            models.Attendance.student_id.in_(student_ids),
            models.Attendance.date >= ninety_days_ago
        ).all()
        
        # Calculate metrics
        metrics = self._calculate_school_metrics(students, results, attendance)
        
        # Performance trends
        trends = self._analyze_performance_trends(results)
        
        # Student segmentation
        segments = self._segment_students(students, results, attendance)
        
        # Predictive analytics
        predictions = self._predict_school_performance(results, attendance)
        
        # Benchmarking (if we had comparison data)
        benchmarks = self._calculate_benchmarks(metrics)
        
        # Generate insights
        insights = self._generate_school_insights(metrics, trends, segments, predictions)
        
        # Recommendations
        recommendations = self._generate_school_recommendations(metrics, segments, insights)
        
        return {
            "school_id": school_id,
            "analysis_date": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "trends": trends,
            "student_segments": segments,
            "predictions": predictions,
            "benchmarks": benchmarks,
            "insights": insights,
            "recommendations": recommendations,
            "visualizations": self._create_school_visualizations(metrics, trends, segments)
        }
    
    def _calculate_school_metrics(
        self, 
        students: List[models.Student],
        results: List[models.ExamResult],
        attendance: List[models.Attendance]
    ) -> Dict[str, Any]:
        """Calculate comprehensive school metrics"""
        metrics = {}
        
        # Academic metrics
        if results:
            scores = [r.percentage for r in results if r.percentage is not None]
            metrics["academic"] = {
                "average_score": round(np.mean(scores), 2) if scores else 0,
                "median_score": round(np.median(scores), 2) if scores else 0,
                "score_std": round(np.std(scores), 2) if len(scores) > 1 else 0,
                "pass_rate": round(len([s for s in scores if s >= 40]) / len(scores) * 100, 2) if scores else 0,
                "excellence_rate": round(len([s for s in scores if s >= 80]) / len(scores) * 100, 2) if scores else 0,
                "failure_rate": round(len([s for s in scores if s < 40]) / len(scores) * 100, 2) if scores else 0,
                "total_exams": len(results),
                "unique_subjects": len(set([r.subject_id for r in results if r.subject_id]))
            }
        
        # Attendance metrics
        if attendance:
            total_days = len(attendance)
            present_days = len([a for a in attendance if a.status == "present"])
            metrics["attendance"] = {
                "attendance_rate": round(present_days / total_days * 100, 2),
                "absent_rate": round(len([a for a in attendance if a.status == "absent"]) / total_days * 100, 2),
                "late_rate": round(len([a for a in attendance if a.status == "late"]) / total_days * 100, 2),
                "average_daily_attendance": present_days / (total_days / len(set([a.student_id for a in attendance]))),
                "chronic_absenteeism": self._calculate_chronic_absenteeism(attendance)
            }
        
        # Student demographics
        if students:
            metrics["demographics"] = {
                "total_students": len(students),
                "gender_distribution": {
                    "male": len([s for s in students if s.gender and s.gender.lower() == "male"]),
                    "female": len([s for s in students if s.gender and s.gender.lower() == "female"]),
                    "other": len([s for s in students if not s.gender or s.gender.lower() not in ["male", "female"]])
                },
                "class_distribution": self._calculate_class_distribution(students),
                "average_age": self._calculate_average_age(students)
            }
        
        # AI Risk metrics
        dropout_risks = [s.dropout_risk_score for s in students if s.dropout_risk_score]
        engagement_scores = [s.engagement_score for s in students if s.engagement_score]
        
        metrics["risk_analysis"] = {
            "average_dropout_risk": round(np.mean(dropout_risks), 3) if dropout_risks else 0,
            "high_risk_students": len([s for s in students if s.dropout_risk_score and s.dropout_risk_score > 0.7]),
            "average_engagement": round(np.mean(engagement_scores), 3) if engagement_scores else 0,
            "disengaged_students": len([s for s in students if s.engagement_score and s.engagement_score < 0.3])
        }
        
        # Efficiency metrics
        if results and attendance:
            metrics["efficiency"] = {
                "academic_efficiency": self._calculate_academic_efficiency(results, attendance),
                "resource_utilization": self._estimate_resource_utilization(students, results),
                "value_added": self._calculate_value_added(results)
            }
        
        return metrics
    
    def _calculate_chronic_absenteeism(self, attendance: List[models.Attendance]) -> Dict[str, Any]:
        """Calculate chronic absenteeism metrics"""
        # Group attendance by student
        student_attendance = {}
        for record in attendance:
            if record.student_id not in student_attendance:
                student_attendance[record.student_id] = []
            student_attendance[record.student_id].append(record)
        
        chronic_absentees = 0
        for student_id, records in student_attendance.items():
            absent_days = len([r for r in records if r.status == "absent"])
            total_days = len(records)
            if total_days > 0 and (absent_days / total_days) > 0.1:  # More than 10% absent
                chronic_absentees += 1
        
        return {
            "chronic_absentee_count": chronic_absentees,
            "chronic_absentee_rate": round(chronic_absentees / len(student_attendance) * 100, 2) if student_attendance else 0,
            "definition": "Students absent >10% of school days"
        }
    
    def _calculate_class_distribution(self, students: List[models.Student]) -> Dict[str, int]:
        """Calculate class level distribution"""
        distribution = {}
        for student in students:
            if student.class_level:
                if student.class_level not in distribution:
                    distribution[student.class_level] = 0
                distribution[student.class_level] += 1
        return distribution
    
    def _calculate_average_age(self, students: List[models.Student]) -> float:
        """Calculate average student age"""
        ages = []
        for student in students:
            if student.date_of_birth:
                today = date.today()
                age = today.year - student.date_of_birth.year
                if (today.month, today.day) < (student.date_of_birth.month, student.date_of_birth.day):
                    age -= 1
                ages.append(age)
        
        return round(np.mean(ages), 1) if ages else 0
    
    def _calculate_academic_efficiency(self, results: List[models.ExamResult], 
                                     attendance: List[models.Attendance]) -> float:
        """Calculate academic efficiency (score per attendance day)"""
        if not results or not attendance:
            return 0
        
        total_score = sum([r.percentage for r in results if r.percentage])
        total_attendance_days = len(set([(a.student_id, a.date) for a in attendance if a.status == "present"]))
        
        return round(total_score / total_attendance_days, 3) if total_attendance_days > 0 else 0
    
    def _estimate_resource_utilization(self, students: List[models.Student], 
                                     results: List[models.ExamResult]) -> Dict[str, float]:
        """Estimate resource utilization metrics"""
        # Simplified estimation
        if not students or not results:
            return {"teacher_student_ratio": 0, "score_per_teacher": 0}
        
        # Assume 1 teacher per 30 students (adjust based on actual data)
        estimated_teachers = max(1, len(students) // 30)
        total_score = sum([r.percentage for r in results if r.percentage])
        
        return {
            "teacher_student_ratio": round(len(students) / estimated_teachers, 1),
            "score_per_teacher": round(total_score / estimated_teachers, 1) if estimated_teachers > 0 else 0,
            "efficiency_index": round(total_score / (len(students) * estimated_teachers), 3)
        }
    
    def _calculate_value_added(self, results: List[models.ExamResult]) -> Dict[str, Any]:
        """Calculate value-added metrics (improvement over time)"""
        if len(results) < 10:
            return {"status": "insufficient_data", "value_added_score": 0}
        
        # Group results by student and calculate improvement
        student_results = {}
        for result in results:
            if result.student_id not in student_results:
                student_results[result.student_id] = []
            student_results[result.student_id].append(result)
        
        improvements = []
        for student_id, student_result_list in student_results.items():
            if len(student_result_list) >= 2:
                # Sort by date
                sorted_results = sorted(student_result_list, 
                                      key=lambda x: x.exam.conducted_date if x.exam else x.created_at)
                first_score = sorted_results[0].percentage
                last_score = sorted_results[-1].percentage
                if first_score and last_score:
                    improvements.append(last_score - first_score)
        
        if improvements:
            return {
                "average_improvement": round(np.mean(improvements), 2),
                "improvement_rate": round(len([i for i in improvements if i > 0]) / len(improvements) * 100, 2),
                "value_added_score": round(np.mean([max(0, i) for i in improvements]), 2),
                "students_improving": len([i for i in improvements if i > 0]),
                "students_declining": len([i for i in improvements if i < 0])
            }
        
        return {"status": "no_improvement_data", "value_added_score": 0}
    
    def _analyze_performance_trends(self, results: List[models.ExamResult]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if not results:
            return {"status": "no_data"}
        
        # Create time series data
        time_series = []
        for result in results:
            if result.exam and result.exam.conducted_date and result.percentage:
                time_series.append({
                    "date": result.exam.conducted_date,
                    "score": result.percentage,
                    "subject": result.subject.name if result.subject else "Unknown"
                })
        
        if len(time_series) < 5:
            return {"status": "insufficient_data", "message": "Need at least 5 data points"}
        
        # Convert to DataFrame
        df = pd.DataFrame(time_series)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Monthly aggregation
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_stats = df.groupby('year_month').agg({
            'score': ['mean', 'std', 'count']
        }).round(2)
        
        # Calculate trend
        X = np.arange(len(monthly_stats)).reshape(-1, 1)
        y = monthly_stats[('score', 'mean')].values
        
        if len(y) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
            trend = "improving" if slope > 0.5 else "declining" if slope < -0.5 else "stable"
        else:
            slope, trend = 0, "insufficient_data"
        
        # Seasonality analysis
        seasonal_patterns = self._detect_seasonal_patterns(df)
        
        # Subject-wise trends
        subject_trends = {}
        for subject in df['subject'].unique():
            subject_data = df[df['subject'] == subject]
            if len(subject_data) >= 3:
                subject_slope, _, _, _, _ = stats.linregress(
                    np.arange(len(subject_data)), 
                    subject_data['score'].values
                )
                subject_trends[subject] = {
                    "trend": "improving" if subject_slope > 0 else "declining",
                    "slope": round(float(subject_slope), 3),
                    "average": round(subject_data['score'].mean(), 2)
                }
        
        return {
            "status": "analyzed",
            "time_period": {
                "start": df['date'].min().strftime('%Y-%m-%d'),
                "end": df['date'].max().strftime('%Y-%m-%d'),
                "total_months": len(monthly_stats)
            },
            "overall_trend": {
                "direction": trend,
                "slope": round(float(slope), 3) if 'slope' in locals() else 0,
                "strength": round(abs(float(slope)) * 10, 2) if 'slope' in locals() else 0
            },
            "monthly_statistics": monthly_stats.to_dict(),
            "seasonal_patterns": seasonal_patterns,
            "subject_trends": subject_trends,
            "volatility": {
                "average_monthly_std": round(monthly_stats[('score', 'std')].mean(), 2),
                "max_volatility_month": monthly_stats[('score', 'std')].idxmax().strftime('%Y-%m'),
                "consistency_score": round(100 - (monthly_stats[('score', 'std')].mean() / 10), 2)
            }
        }
    
    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in performance"""
        patterns = {}
        
        # Monthly patterns
        df['month'] = df['date'].dt.month
        monthly_avg = df.groupby('month')['score'].mean()
        
        if len(monthly_avg) >= 3:
            best_month = monthly_avg.idxmax()
            worst_month = monthly_avg.idxmin()
            
            patterns["monthly"] = {
                "best_performing_month": int(best_month),
                "best_month_score": round(float(monthly_avg.max()), 2),
                "worst_performing_month": int(worst_month),
                "worst_month_score": round(float(monthly_avg.min()), 2),
                "seasonal_variation": round(float(monthly_avg.max() - monthly_avg.min()), 2)
            }
        
        # Term patterns (assuming terms: Jan-Apr, May-Aug, Sep-Dec)
        def get_term(month):
            if 1 <= month <= 4:
                return "Term 1"
            elif 5 <= month <= 8:
                return "Term 2"
            else:
                return "Term 3"
        
        df['term'] = df['month'].apply(get_term)
        term_avg = df.groupby('term')['score'].mean()
        
        if len(term_avg) >= 2:
            patterns["term"] = {
                "best_term": term_avg.idxmax(),
                "best_term_score": round(float(term_avg.max()), 2),
                "performance_by_term": term_avg.round(2).to_dict()
            }
        
        return patterns
    
    def _segment_students(
        self, 
        students: List[models.Student],
        results: List[models.ExamResult],
        attendance: List[models.Attendance]
    ) -> Dict[str, Any]:
        """Segment students into clusters based on multiple factors"""
        if not students:
            return {"status": "no_students"}
        
        # Prepare feature matrix
        features = []
        student_ids = []
        
        for student in students:
            student_results = [r for r in results if r.student_id == student.id]
            student_attendance = [a for a in attendance if a.student_id == student.id]
            
            # Extract features
            feature_vector = self._extract_student_features(student, student_results, student_attendance)
            features.append(feature_vector)
            student_ids.append(student.id)
        
        if len(features) < 5:
            return {"status": "insufficient_data_for_clustering"}
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine optimal number of clusters
        optimal_k = self._determine_optimal_clusters(features_scaled)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=min(optimal_k, len(features_scaled)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(clusters, students, results, attendance, features)
        
        return {
            "status": "segmented",
            "number_of_clusters": int(optimal_k),
            "clustering_method": "K-means with standardized features",
            "cluster_distribution": dict(zip(*np.unique(clusters, return_counts=True))),
            "cluster_profiles": cluster_analysis,
            "student_assignments": [
                {"student_id": student_ids[i], "cluster": int(clusters[i])}
                for i in range(len(student_ids))
            ]
        }
    
    def _extract_student_features(
        self,
        student: models.Student,
        results: List[models.ExamResult],
        attendance: List[models.Attendance]
    ) -> List[float]:
        """Extract features for student clustering"""
        features = []
        
        # Academic features
        if results:
            scores = [r.percentage for r in results if r.percentage]
            features.extend([
                np.mean(scores) if scores else 0,  # Average score
                np.std(scores) if len(scores) > 1 else 0,  # Score consistency
                len(scores)  # Number of exams
            ])
        else:
            features.extend([0, 0, 0])
        
        # Attendance features
        if attendance:
            present_days = len([a for a in attendance if a.status == "present"])
            total_days = len(attendance)
            features.extend([
                present_days / total_days if total_days > 0 else 0,  # Attendance rate
                len([a for a in attendance if a.status == "absent"])  # Absence count
            ])
        else:
            features.extend([0, 0])
        
        # Demographic features
        features.extend([
            student.dropout_risk_score if student.dropout_risk_score else 0,
            student.engagement_score if student.engagement_score else 0,
            1 if student.gender and student.gender.lower() == "male" else 0,
            self._calculate_age(student.date_of_birth) if student.date_of_birth else 18
        ])
        
        return features
    
    def _calculate_age(self, dob: date) -> float:
        """Calculate age from date of birth"""
        today = date.today()
        age = today.year - dob.year
        if (today.month, today.day) < (dob.month, dob.day):
            age -= 1
        return age
    
    def _determine_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Determine optimal number of clusters using elbow method"""
        if len(data) <= 3:
            return min(2, len(data))
        
        max_k = min(max_k, len(data) - 1)
        inertias = []
        K = range(2, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection (find point of maximum curvature)
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diff_ratios = diffs[1:] / diffs[:-1]
            optimal_k = K[np.argmin(diff_ratios) + 1]
        else:
            optimal_k = 3
        
        return min(optimal_k, len(data) // 3)  # Ensure reasonable cluster size
    
    def _analyze_clusters(
        self,
        clusters: np.ndarray,
        students: List[models.Student],
        results: List[models.ExamResult],
        attendance: List[models.Attendance],
        features: List[List[float]]
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of each cluster"""
        cluster_analysis = {}
        
        for cluster_id in np.unique(clusters):
            cluster_students = [students[i] for i in range(len(students)) if clusters[i] == cluster_id]
            cluster_features = [features[i] for i in range(len(features)) if clusters[i] == cluster_id]
            
            if not cluster_students:
                continue
            
            # Basic statistics
            cluster_scores = []
            cluster_attendance = []
            
            for student in cluster_students:
                student_results = [r for r in results if r.student_id == student.id]
                student_attendance = [a for a in attendance if a.student_id == student.id]
                
                if student_results:
                    scores = [r.percentage for r in student_results if r.percentage]
                    if scores:
                        cluster_scores.extend(scores)
                
                if student_attendance:
                    present_days = len([a for a in student_attendance if a.status == "present"])
                    total_days = len(student_attendance)
                    if total_days > 0:
                        cluster_attendance.append(present_days / total_days)
            
            # Feature analysis
            feature_means = np.mean(cluster_features, axis=0) if cluster_features else []
            
            cluster_analysis[int(cluster_id)] = {
                "size": len(cluster_students),
                "percentage": round(len(cluster_students) / len(students) * 100, 2),
                "academic_performance": {
                    "average_score": round(np.mean(cluster_scores), 2) if cluster_scores else 0,
                    "score_range": [round(min(cluster_scores), 2), round(max(cluster_scores), 2)] if cluster_scores else [0, 0],
                    "consistency": round(np.std(cluster_scores), 2) if len(cluster_scores) > 1 else 0
                },
                "attendance": {
                    "average_rate": round(np.mean(cluster_attendance) * 100, 2) if cluster_attendance else 0,
                    "range": [round(min(cluster_attendance) * 100, 2), round(max(cluster_attendance) * 100, 2)] if cluster_attendance else [0, 0]
                },
                "risk_profile": {
                    "average_dropout_risk": round(np.mean([s.dropout_risk_score for s in cluster_students if s.dropout_risk_score]), 3),
                    "high_risk_count": len([s for s in cluster_students if s.dropout_risk_score and s.dropout_risk_score > 0.7])
                },
                "demographics": {
                    "gender_distribution": {
                        "male": len([s for s in cluster_students if s.gender and s.gender.lower() == "male"]),
                        "female": len([s for s in cluster_students if s.gender and s.gender.lower() == "female"])
                    },
                    "average_age": round(np.mean([self._calculate_age(s.date_of_birth) 
                                                for s in cluster_students if s.date_of_birth]), 1)
                },
                "cluster_label": self._assign_cluster_label(feature_means),
                "key_characteristics": self._identify_key_characteristics(feature_means),
                "recommended_interventions": self._suggest_cluster_interventions(feature_means)
            }
        
        return cluster_analysis
    
    def _assign_cluster_label(self, feature_means: List[float]) -> str:
        """Assign descriptive label to cluster based on features"""
        if len(feature_means) < 7:
            return "General"
        
        # Feature indices (based on _extract_student_features)
        # 0: average_score, 1: score_std, 2: exam_count, 3: attendance_rate, 
        # 4: absence_count, 5: dropout_risk, 6: engagement
        
        avg_score = feature_means[0]
        attendance = feature_means[3]
        dropout_risk = feature_means[5]
        engagement = feature_means[6]
        
        if avg_score > 70 and attendance > 0.8:
            return "High Achievers"
        elif avg_score < 40 and dropout_risk > 0.6:
            return "At-Risk Students"
        elif attendance < 0.6 and engagement < 0.4:
            return "Disengaged Learners"
        elif avg_score > 60 and engagement > 0.7:
            return "Engaged Performers"
        elif 40 <= avg_score <= 60 and 0.5 <= attendance <= 0.7:
            return "Average Students"
        else:
            return "Mixed Profile"
    
    def _identify_key_characteristics(self, feature_means: List[float]) -> List[str]:
        """Identify key characteristics of cluster"""
        characteristics = []
        
        if len(feature_means) < 7:
            return ["Insufficient data"]
        
        avg_score = feature_means[0]
        score_std = feature_means[1]
        attendance = feature_means[3]
        dropout_risk = feature_means[5]
        engagement = feature_means[6]
        
        if avg_score > 75:
            characteristics.append("High academic performance")
        elif avg_score < 45:
            characteristics.append("Low academic performance")
        
        if score_std > 15:
            characteristics.append("Inconsistent performance")
        elif score_std < 5:
            characteristics.append("Consistent performance")
        
        if attendance > 0.85:
            characteristics.append("Excellent attendance")
        elif attendance < 0.6:
            characteristics.append("Poor attendance")
        
        if dropout_risk > 0.7:
            characteristics.append("High dropout risk")
        elif dropout_risk < 0.3:
            characteristics.append("Low dropout risk")
        
        if engagement > 0.8:
            characteristics.append("Highly engaged")
        elif engagement < 0.4:
            characteristics.append("Low engagement")
        
        return characteristics[:3]  # Return top 3
    
    def _suggest_cluster_interventions(self, feature_means: List[float]) -> List[str]:
        """Suggest interventions for cluster"""
        interventions = []
        
        if len(feature_means) < 7:
            return ["Collect more data for personalized recommendations"]
        
        avg_score = feature_means[0]
        attendance = feature_means[3]
        dropout_risk = feature_means[5]
        engagement = feature_means[6]
        
        if avg_score < 50:
            interventions.append("Targeted academic support programs")
        
        if attendance < 0.7:
            interventions.append("Attendance improvement initiatives")
        
        if dropout_risk > 0.6:
            interventions.append("Dropout prevention interventions")
        
        if engagement < 0.5:
            interventions.append("Engagement enhancement activities")
        
        if avg_score > 70 and engagement > 0.7:
            interventions.append("Advanced learning opportunities")
        
        return interventions[:3]
    
    def _predict_school_performance(
        self, 
        results: List[models.ExamResult],
        attendance: List[models.Attendance]
    ) -> Dict[str, Any]:
        """Predict future school performance"""
        if not results or len(results) < 10:
            return {"status": "insufficient_data", "message": "Need more historical data"}
        
        # Prepare time series data
        time_series = []
        for result in results:
            if result.exam and result.exam.conducted_date and result.percentage:
                time_series.append({
                    "date": result.exam.conducted_date,
                    "score": result.percentage
                })
        
        if len(time_series) < 10:
            return {"status": "insufficient_time_series"}
        
        df = pd.DataFrame(time_series)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Aggregate by month
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('year_month')['score'].mean().reset_index()
        monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
        
        # Use simple forecasting (could use Prophet, ARIMA, etc. in production)
        try:
            # Linear regression forecast for next 3 months
            X = np.arange(len(monthly)).reshape(-1, 1)
            y = monthly['score'].values
            
            model = sm.OLS(y, sm.add_constant(X))
            results_model = model.fit()
            
            # Predict next 3 months
            future_X = np.array([[len(monthly)], [len(monthly) + 1], [len(monthly) + 2]])
            predictions = results_model.predict(sm.add_constant(future_X))
            
            # Calculate confidence intervals
            predictions_df = results_model.get_prediction(sm.add_constant(future_X))
            predictions_summary = predictions_df.summary_frame(alpha=0.05)
            
            forecast = []
            for i in range(3):
                month = (monthly['year_month'].iloc[-1] + pd.DateOffset(months=i+1))
                forecast.append({
                    "month": month.strftime('%Y-%m'),
                    "predicted_score": round(float(predictions[i]), 2),
                    "lower_bound": round(float(predictions_summary['mean_ci_lower'].iloc[i]), 2),
                    "upper_bound": round(float(predictions_summary['mean_ci_upper'].iloc[i]), 2),
                    "confidence": round(100 - (predictions_summary['mean_ci_upper'].iloc[i] - 
                                             predictions_summary['mean_ci_lower'].iloc[i]), 2)
                })
            
            # Calculate growth projection
            current_avg = monthly['score'].iloc[-3:].mean()
            projected_avg = np.mean(predictions)
            growth = ((projected_avg - current_avg) / current_avg) * 100 if current_avg > 0 else 0
            
            # Risk assessment
            risk_factors = []
            if current_avg < 50:
                risk_factors.append("Low current performance")
            if growth < 0:
                risk_factors.append("Negative growth projected")
            if len([p for p in predictions if p < 40]) > 0:
                risk_factors.append("Risk of falling below passing threshold")
            
            return {
                "status": "forecast_generated",
                "forecast_horizon": "3 months",
                "current_performance": round(float(current_avg), 2),
                "projected_performance": round(float(projected_avg), 2),
                "projected_growth": round(float(growth), 2),
                "monthly_forecast": forecast,
                "confidence_level": "medium",  # Based on model R-squared
                "risk_factors": risk_factors,
                "recommendations": self._generate_forecast_recommendations(current_avg, projected_avg, growth, risk_factors)
            }
            
        except Exception as e:
            return {
                "status": "forecast_failed",
                "error": str(e),
                "fallback_prediction": {
                    "message": "Using simple average projection",
                    "projected_score": round(df['score'].mean(), 2)
                }
            }
    
    def _generate_forecast_recommendations(
        self, 
        current_avg: float, 
        projected_avg: float, 
        growth: float,
        risk_factors: List[str]
    ) -> List[str]:
        """Generate recommendations based on forecast"""
        recommendations = []
        
        if current_avg < 50:
            recommendations.append("Implement immediate academic improvement plan")
        
        if projected_avg < current_avg:
            recommendations.append("Review and adjust current teaching strategies")
        
        if growth > 5:
            recommendations.append("Continue current successful practices")
        elif growth < 0:
            recommendations.append("Investigate causes of projected decline")
        
        if "Low current performance" in risk_factors:
            recommendations.append("Prioritize support for struggling students")
        
        if "Risk of falling below passing threshold" in risk_factors:
            recommendations.append("Implement targeted interventions for at-risk students")
        
        if not recommendations:
            recommendations.append("Maintain current performance monitoring")
        
        return recommendations[:3]
    
    def _calculate_benchmarks(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance benchmarks"""
        # These would typically come from external data or historical averages
        # For now, using hypothetical benchmarks
        
        benchmarks = {
            "academic": {
                "national_average": 55.0,  # Hypothetical
                "county_average": 58.0,    # Hypothetical
                "excellence_threshold": 75.0,
                "passing_threshold": 40.0
            },
            "attendance": {
                "national_average": 85.0,
                "acceptable_threshold": 75.0,
                "excellent_threshold": 95.0
            },
            "efficiency": {
                "target_teacher_student_ratio": 30.0,
                "target_score_per_student": 60.0
            }
        }
        
        # Calculate performance against benchmarks
        performance_vs_benchmarks = {}
        
        if "academic" in metrics:
            performance_vs_benchmarks["academic"] = {
                "vs_national": round(metrics["academic"]["average_score"] - benchmarks["academic"]["national_average"], 2),
                "vs_county": round(metrics["academic"]["average_score"] - benchmarks["academic"]["county_average"], 2),
                "status": "above" if metrics["academic"]["average_score"] > benchmarks["academic"]["county_average"] else "below",
                "percentile_estimate": self._estimate_percentile(metrics["academic"]["average_score"], "academic")
            }
        
        if "attendance" in metrics:
            performance_vs_benchmarks["attendance"] = {
                "vs_national": round(metrics["attendance"]["attendance_rate"] - benchmarks["attendance"]["national_average"], 2),
                "status": "excellent" if metrics["attendance"]["attendance_rate"] > benchmarks["attendance"]["excellent_threshold"] 
                         else "acceptable" if metrics["attendance"]["attendance_rate"] > benchmarks["attendance"]["acceptable_threshold"]
                         else "needs_improvement",
                "ranking": self._estimate_ranking(metrics["attendance"]["attendance_rate"], "attendance")
            }
        
        return {
            "benchmark_values": benchmarks,
            "performance_vs_benchmarks": performance_vs_benchmarks,
            "overall_benchmark_score": self._calculate_overall_benchmark_score(metrics, benchmarks)
        }
    
    def _estimate_percentile(self, score: float, metric_type: str) -> float:
        """Estimate percentile ranking (simplified)"""
        # In production, this would use actual distribution data
        if metric_type == "academic":
            if score >= 80:
                return 90.0
            elif score >= 70:
                return 75.0
            elif score >= 60:
                return 50.0
            elif score >= 50:
                return 30.0
            else:
                return 15.0
        elif metric_type == "attendance":
            if score >= 95:
                return 95.0
            elif score >= 90:
                return 80.0
            elif score >= 80:
                return 60.0
            elif score >= 70:
                return 40.0
            else:
                return 20.0
        return 50.0
    
    def _estimate_ranking(self, value: float, metric_type: str) -> str:
        """Estimate ranking category"""
        if metric_type == "academic":
            if value >= 75:
                return "Top 25%"
            elif value >= 60:
                return "Above Average"
            elif value >= 45:
                return "Average"
            else:
                return "Below Average"
        elif metric_type == "attendance":
            if value >= 95:
                return "Excellent"
            elif value >= 85:
                return "Good"
            elif value >= 75:
                return "Fair"
            else:
                return "Poor"
        return "Average"
    
    def _calculate_overall_benchmark_score(self, metrics: Dict[str, Any], benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall benchmark score"""
        scores = []
        weights = {"academic": 0.5, "attendance": 0.3, "efficiency": 0.2}
        
        if "academic" in metrics:
            academic_score = min(100, (metrics["academic"]["average_score"] / benchmarks["academic"]["county_average"]) * 100)
            scores.append(academic_score * weights["academic"])
        
        if "attendance" in metrics:
            attendance_score = min(100, (metrics["attendance"]["attendance_rate"] / benchmarks["attendance"]["national_average"]) * 100)
            scores.append(attendance_score * weights["attendance"])
        
        if "efficiency" in metrics:
            efficiency_score = 80  # Placeholder
            scores.append(efficiency_score * weights["efficiency"])
        
        overall_score = round(sum(scores), 2) if scores else 0
        
        return {
            "overall_score": overall_score,
            "rating": self._get_rating(overall_score),
            "component_scores": {
                "academic": round(academic_score, 2) if 'academic_score' in locals() else 0,
                "attendance": round(attendance_score, 2) if 'attendance_score' in locals() else 0,
                "efficiency": round(efficiency_score, 2) if 'efficiency_score' in locals() else 0
            },
            "improvement_areas": self._identify_improvement_areas(metrics, benchmarks)
        }
    
    def _get_rating(self, score: float) -> str:
        """Get rating from score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Satisfactory"
        elif score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _identify_improvement_areas(self, metrics: Dict[str, Any], benchmarks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas needing improvement"""
        improvement_areas = []
        
        if "academic" in metrics:
            if metrics["academic"]["average_score"] < benchmarks["academic"]["county_average"]:
                improvement_areas.append({
                    "area": "Academic Performance",
                    "current": metrics["academic"]["average_score"],
                    "target": benchmarks["academic"]["county_average"],
                    "gap": round(benchmarks["academic"]["county_average"] - metrics["academic"]["average_score"], 2),
                    "priority": "High" if metrics["academic"]["average_score"] < 50 else "Medium"
                })
        
        if "attendance" in metrics:
            if metrics["attendance"]["attendance_rate"] < benchmarks["attendance"]["acceptable_threshold"]:
                improvement_areas.append({
                    "area": "Attendance",
                    "current": metrics["attendance"]["attendance_rate"],
                    "target": benchmarks["attendance"]["acceptable_threshold"],
                    "gap": round(benchmarks["attendance"]["acceptable_threshold"] - metrics["attendance"]["attendance_rate"], 2),
                    "priority": "High"
                })
        
        # Sort by priority and gap
        improvement_areas.sort(key=lambda x: (x["priority"] == "High", -x["gap"]), reverse=True)
        
        return improvement_areas[:3]  # Return top 3
    
    def _generate_school_insights(
        self, 
        metrics: Dict[str, Any], 
        trends: Dict[str, Any], 
        segments: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered insights for school"""
        insights = []
        
        # Academic insights
        if "academic" in metrics:
            if metrics["academic"]["average_score"] > 70:
                insights.append({
                    "type": "strength",
                    "category": "academic",
                    "title": "Strong Academic Performance",
                    "description": f"School maintains above-average academic performance ({metrics['academic']['average_score']}%)",
                    "impact": "High",
                    "evidence": f"Average score is {round(metrics['academic']['average_score'] - 55, 2)}% above national average"
                })
            elif metrics["academic"]["average_score"] < 45:
                insights.append({
                    "type": "concern",
                    "category": "academic",
                    "title": "Academic Performance Needs Attention",
                    "description": f"Academic performance below acceptable levels ({metrics['academic']['average_score']}%)",
                    "impact": "Critical",
                    "evidence": f"{metrics['academic']['failure_rate']}% of students are failing"
                })
        
        # Attendance insights
        if "attendance" in metrics:
            if metrics["attendance"]["attendance_rate"] > 90:
                insights.append({
                    "type": "strength",
                    "category": "attendance",
                    "title": "Excellent Student Attendance",
                    "description": f"School maintains excellent attendance rate ({metrics['attendance']['attendance_rate']}%)",
                    "impact": "Medium",
                    "evidence": f"Only {metrics['attendance']['absent_rate']}% absent rate"
                })
            elif metrics["attendance"]["attendance_rate"] < 75:
                insights.append({
                    "type": "concern",
                    "category": "attendance",
                    "title": "Attendance Issues Detected",
                    "description": f"Attendance rate below acceptable threshold ({metrics['attendance']['attendance_rate']}%)",
                    "impact": "High",
                    "evidence": f"{metrics['attendance']['chronic_absenteeism']['chronic_absentee_count']} chronic absentees identified"
                })
        
        # Trend insights
        if trends.get("status") == "analyzed":
            trend_direction = trends.get("overall_trend", {}).get("direction", "stable")
            if trend_direction == "improving":
                insights.append({
                    "type": "opportunity",
                    "category": "trend",
                    "title": "Positive Performance Trend",
                    "description": "School performance shows consistent improvement over time",
                    "impact": "High",
                    "evidence": f"Monthly improvement rate: {trends['overall_trend']['slope']}"
                })
            elif trend_direction == "declining":
                insights.append({
                    "type": "risk",
                    "category": "trend",
                    "title": "Declining Performance Trend",
                    "description": "School performance shows concerning decline",
                    "impact": "Critical",
                    "evidence": f"Monthly decline rate: {abs(trends['overall_trend']['slope'])}"
                })
        
        # Segment insights
        if segments.get("status") == "segmented":
            cluster_dist = segments.get("cluster_distribution", {})
            if len(cluster_dist) > 0:
                largest_cluster = max(cluster_dist.items(), key=lambda x: x[1])
                cluster_label = segments.get("cluster_profiles", {}).get(str(largest_cluster[0]), {}).get("cluster_label", "Unknown")
                
                insights.append({
                    "type": "pattern",
                    "category": "student_segmentation",
                    "title": f"Dominant Student Profile: {cluster_label}",
                    "description": f"{largest_cluster[1]} students ({largest_cluster[1]/sum(cluster_dist.values())*100:.1f}%) share similar characteristics",
                    "impact": "Medium",
                    "evidence": f"Cluster size: {largest_cluster[1]} students"
                })
        
        # Prediction insights
        if predictions.get("status") == "forecast_generated":
            if predictions.get("projected_growth", 0) > 5:
                insights.append({
                    "type": "opportunity",
                    "category": "forecast",
                    "title": "Positive Growth Forecast",
                    "description": f"Projected {predictions['projected_growth']:.1f}% improvement in next quarter",
                    "impact": "High",
                    "evidence": f"Current: {predictions['current_performance']}%, Projected: {predictions['projected_performance']}%"
                })
            elif predictions.get("projected_growth", 0) < 0:
                insights.append({
                    "type": "risk",
                    "category": "forecast",
                    "title": "Negative Growth Forecast",
                    "description": f"Projected {abs(predictions['projected_growth']):.1f}% decline in next quarter",
                    "impact": "High",
                    "evidence": f"Current: {predictions['current_performance']}%, Projected: {predictions['projected_performance']}%"
                })
        
        return insights[:5]  # Return top 5 insights
    
    def _generate_school_recommendations(
        self, 
        metrics: Dict[str, Any], 
        segments: Dict[str, Any], 
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations for school"""
        recommendations = []
        
        # Academic recommendations
        if "academic" in metrics:
            if metrics["academic"]["average_score"] < 50:
                recommendations.append({
                    "category": "academic_improvement",
                    "priority": "High",
                    "action": "Implement comprehensive academic support program",
                    "rationale": f"Average score ({metrics['academic']['average_score']}%) below acceptable threshold",
                    "expected_impact": "Increase average score by 10-15% within 6 months",
                    "key_activities": [
                        "Diagnostic assessments for all students",
                        "Targeted tutoring for struggling students",
                        "Teacher training on differentiated instruction"
                    ],
                    "success_metrics": ["Average score > 55%", "Failure rate < 20%"],
                    "timeline": "6 months",
                    "resources_needed": ["Tutors", "Assessment materials", "Training budget"]
                })
            
            if metrics["academic"]["excellence_rate"] < 20:
                recommendations.append({
                    "category": "excellence_promotion",
                    "priority": "Medium",
                    "action": "Develop gifted and talented program",
                    "rationale": f"Only {metrics['academic']['excellence_rate']}% of students achieving excellence",
                    "expected_impact": "Double excellence rate within 1 year",
                    "key_activities": [
                        "Identify high-potential students",
                        "Provide advanced learning opportunities",
                        "Establish mentorship program"
                    ],
                    "success_metrics": ["Excellence rate > 30%", "Top student satisfaction > 80%"],
                    "timeline": "12 months",
                    "resources_needed": ["Advanced curriculum", "Mentors", "Enrichment materials"]
                })
        
        # Attendance recommendations
        if "attendance" in metrics:
            if metrics["attendance"]["attendance_rate"] < 80:
                recommendations.append({
                    "category": "attendance_improvement",
                    "priority": "High",
                    "action": "Launch attendance improvement campaign",
                    "rationale": f"Attendance rate ({metrics['attendance']['attendance_rate']}%) below target",
                    "expected_impact": "Increase attendance rate to 85% within 3 months",
                    "key_activities": [
                        "Daily attendance monitoring",
                        "Parent engagement program",
                        "Incentives for perfect attendance"
                    ],
                    "success_metrics": ["Attendance rate > 85%", "Chronic absenteeism < 5%"],
                    "timeline": "3 months",
                    "resources_needed": ["Monitoring system", "Incentive budget", "Parent communication tools"]
                })
        
        # Segment-based recommendations
        if segments.get("status") == "segmented":
            cluster_profiles = segments.get("cluster_profiles", {})
            for cluster_id, profile in cluster_profiles.items():
                if profile.get("size", 0) > 10:  # Significant cluster
                    cluster_label = profile.get("cluster_label", "")
                    
                    if "At-Risk" in cluster_label:
                        recommendations.append({
                            "category": "targeted_intervention",
                            "priority": "High",
                            "action": f"Targeted support for {cluster_label} cluster",
                            "rationale": f"{profile['size']} students identified as high-risk",
                            "expected_impact": "Reduce dropout risk by 50% within 6 months",
                            "key_activities": [
                                "Individualized support plans",
                                "Regular counseling sessions",
                                "Parent partnership program"
                            ],
                            "success_metrics": [f"Cluster dropout risk < 0.3", "Attendance improvement > 15%"],
                            "timeline": "6 months",
                            "resources_needed": ["Counselors", "Support staff", "Family engagement resources"]
                        })
        
        # Insight-based recommendations
        critical_insights = [i for i in insights if i.get("impact") == "Critical"]
        for insight in critical_insights[:2]:  # Top 2 critical insights
            recommendations.append({
                "category": "urgent_response",
                "priority": "Critical",
                "action": f"Immediate response to {insight['title']}",
                "rationale": insight["description"],
                "expected_impact": "Mitigate critical risk within 30 days",
                "key_activities": [
                    "Form emergency response team",
                    "Implement immediate corrective actions",
                    "Daily progress monitoring"
                ],
                "success_metrics": ["Risk mitigated within 30 days", "No further deterioration"],
                "timeline": "30 days",
                "resources_needed": ["Emergency budget", "Task force", "Monitoring tools"]
            })
        
        # Sort by priority
        priority_order = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _create_school_visualizations(
        self, 
        metrics: Dict[str, Any], 
        trends: Dict[str, Any], 
        segments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create visualization data for school dashboard"""
        visualizations = {}
        
        # Performance trend chart
        if trends.get("status") == "analyzed" and "monthly_statistics" in trends:
            monthly_data = trends["monthly_statistics"]
            visualizations["performance_trend"] = {
                "type": "line_chart",
                "title": "Academic Performance Trend",
                "data": {
                    "labels": [str(idx) for idx in monthly_data[('score', 'mean')].keys()],
                    "datasets": [
                        {
                            "label": "Average Score",
                            "data": [float(v) for v in monthly_data[('score', 'mean')].values()],
                            "borderColor": "rgb(75, 192, 192)",
                            "backgroundColor": "rgba(75, 192, 192, 0.2)"
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False
                }
            }
        
        # Student segmentation chart
        if segments.get("status") == "segmented" and "cluster_distribution" in segments:
            cluster_data = segments["cluster_distribution"]
            visualizations["student_segmentation"] = {
                "type": "doughnut_chart",
                "title": "Student Segmentation",
                "data": {
                    "labels": [f"Cluster {k}" for k in cluster_data.keys()],
                    "datasets": [
                        {
                            "data": list(cluster_data.values()),
                            "backgroundColor": [
                                "rgb(255, 99, 132)",
                                "rgb(54, 162, 235)",
                                "rgb(255, 205, 86)",
                                "rgb(75, 192, 192)",
                                "rgb(153, 102, 255)"
                            ]
                        }
                    ]
                }
            }
        
        # Metric comparison radar chart
        if "academic" in metrics and "attendance" in metrics:
            visualizations["metric_radar"] = {
                "type": "radar_chart",
                "title": "Performance Metrics",
                "data": {
                    "labels": ["Academic", "Attendance", "Engagement", "Efficiency", "Safety"],
                    "datasets": [
                        {
                            "label": "Current Performance",
                            "data": [
                                min(100, metrics["academic"]["average_score"]),
                                metrics["attendance"]["attendance_rate"],
                                metrics.get("risk_analysis", {}).get("average_engagement", 50) * 100,
                                75,  # Placeholder for efficiency
                                85   # Placeholder for safety
                            ],
                            "backgroundColor": "rgba(54, 162, 235, 0.2)",
                            "borderColor": "rgb(54, 162, 235)"
                        }
                    ]
                }
            }
        
        # Risk distribution histogram
        if "risk_analysis" in metrics:
            # Simplified risk distribution
            risk_levels = ["Low", "Medium", "High", "Critical"]
            risk_counts = [30, 40, 20, 10]  # Placeholder - would calculate from actual data
            
            visualizations["risk_distribution"] = {
                "type": "bar_chart",
                "title": "Student Risk Distribution",
                "data": {
                    "labels": risk_levels,
                    "datasets": [
                        {
                            "label": "Number of Students",
                            "data": risk_counts,
                            "backgroundColor": [
                                "rgb(75, 192, 192)",
                                "rgb(255, 205, 86)",
                                "rgb(255, 159, 64)",
                                "rgb(255, 99, 132)"
                            ]
                        }
                    ]
                }
            }
        
        return visualizations
    
    def analyze_student_comparison(self, student_ids: List[int]) -> Dict[str, Any]:
        """Compare multiple students"""
        comparisons = []
        
        for student_id in student_ids:
            analysis = self.ai_engine.analyze_student_comprehensive(student_id)
            comparisons.append({
                "student_id": student_id,
                "analysis": analysis
            })
        
        # Generate comparative insights
        comparative_insights = self._generate_comparative_insights(comparisons)
        
        return {
            "comparison_id": f"COMP_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "students_compared": len(student_ids),
            "individual_analyses": comparisons,
            "comparative_insights": comparative_insights,
            "recommendations": self._generate_comparative_recommendations(comparisons)
        }
    
    def _generate_comparative_insights(self, comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights from student comparisons"""
        insights = []
        
        if len(comparisons) < 2:
            return [{"message": "Need at least 2 students for comparison"}]
        
        # Extract key metrics
        dropout_risks = []
        performance_scores = []
        engagement_scores = []
        
        for comp in comparisons:
            analysis = comp["analysis"]
            dropout_risks.append(analysis.get("dropout_risk_analysis", {}).get("risk_score", 0))
            performance_scores.append(analysis.get("performance_analysis", {}).get("current_average", 0))
            engagement_scores.append(analysis.get("engagement_analysis", {}).get("engagement_score", 0))
        
        # Generate insights
        if max(dropout_risks) - min(dropout_risks) > 0.3:
            high_risk_idx = dropout_risks.index(max(dropout_risks))
            insights.append({
                "type": "risk_disparity",
                "title": "Significant Dropout Risk Disparity",
                "description": f"Student {comparisons[high_risk_idx]['student_id']} has {max(dropout_risks):.1%} dropout risk vs {min(dropout_risks):.1%} for others",
                "recommendation": "Prioritize interventions for high-risk student"
            })
        
        if max(performance_scores) - min(performance_scores) > 20:
            insights.append({
                "type": "performance_gap",
                "title": "Large Academic Performance Gap",
                "description": f"Performance gap of {max(performance_scores) - min(performance_scores):.1f}% between students",
                "recommendation": "Implement peer tutoring between high and low performers"
            })
        
        return insights
    
    def _generate_comparative_recommendations(self, comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on student comparisons"""
        recommendations = []
        
        if len(comparisons) >= 2:
            recommendations.append({
                "category": "peer_learning",
                "action": "Establish peer learning pairs based on complementary strengths",
                "rationale": "Students show different strengths and weaknesses",
                "expected_benefit": "Mutual skill development and improved engagement"
            })
            
            recommendations.append({
                "category": "group_intervention",
                "action": "Create targeted intervention group for similar risk profiles",
                "rationale": "Multiple students share similar risk factors",
                "expected_benefit": "Efficient resource utilization and peer support"
            })
        
        return recommendations
    
    def generate_performance_report(self, entity_type: str, entity_id: int, 
                                  start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report_data = {}
        
        if entity_type == "school":
            report_data = self.analyze_school_performance(entity_id)
        elif entity_type == "student":
            report_data = self.ai_engine.analyze_student_comprehensive(entity_id)
        elif entity_type == "class":
            # Implement class analysis
            pass
        
        return {
            "report_id": f"REPORT_{entity_type}_{entity_id}_{datetime.utcnow().strftime('%Y%m%d')}",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat(),
            "data": report_data,
            "summary": self._generate_report_summary(report_data),
            "key_findings": self._extract_key_findings(report_data),
            "action_items": self._generate_action_items(report_data)
        }
    
    def _generate_report_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for report"""
        summary = {
            "overall_status": "Good",
            "key_metrics": {},
            "top_strengths": [],
            "top_concerns": [],
            "recommendation_summary": ""
        }
        
        # Extract from school analysis
        if "metrics" in report_data:
            metrics = report_data["metrics"]
            if "academic" in metrics:
                summary["key_metrics"]["academic_performance"] = metrics["academic"]["average_score"]
            if "attendance" in metrics:
                summary["key_metrics"]["attendance_rate"] = metrics["attendance"]["attendance_rate"]
        
        # Extract from student analysis
        elif "dropout_risk_analysis" in report_data:
            risk = report_data["dropout_risk_analysis"]
            summary["key_metrics"]["dropout_risk"] = risk.get("risk_score", 0)
            summary["key_metrics"]["risk_level"] = risk.get("risk_level", "UNKNOWN")
        
        return summary
    
    def _extract_key_findings(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key findings from report data"""
        findings = []
        
        if "insights" in report_data:
            findings.extend(report_data["insights"][:3])
        
        if "dropout_risk_analysis" in report_data:
            risk = report_data["dropout_risk_analysis"]
            if risk.get("risk_level") in ["HIGH", "CRITICAL"]:
                findings.append({
                    "type": "critical_finding",
                    "title": "High Dropout Risk",
                    "description": f"Dropout risk score: {risk.get('risk_score', 0):.1%}",
                    "priority": "Critical"
                })
        
        return findings[:5]  # Return top 5 findings
    
    def _generate_action_items(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate action items from report"""
        actions = []
        
        if "recommendations" in report_data:
            for rec in report_data["recommendations"][:3]:
                actions.append({
                    "action": rec.get("action", ""),
                    "priority": rec.get("priority", "Medium"),
                    "timeline": rec.get("timeline", "30 days"),
                    "owner": "School Administration"  # Would be assigned based on context
                })
        
        return actions
    
    def close(self):
        """Close database connection"""
        self.db.close()
        self.ai_engine.close()
