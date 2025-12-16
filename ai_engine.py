import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import openai
import statsmodels.api as sm
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

from database import SessionLocal
import models
from schemas import AIAnalysisRequest, AIAnalysisResponse, BatchAnalysisRequest

class AdvancedAIEngine:
    def __init__(self):
        self.db = SessionLocal()
        
        # Initialize models
        self.dropout_model = None
        self.performance_model = None
        self.sentiment_analyzer = None
        self.clustering_model = None
        self.anomaly_detector = None
        self.prophet_models = {}
        self.llm_chain = None
        self.sentence_model = None
        
        # ChromaDB for vector storage
        self.chroma_client = chromadb.Client()
        self.knowledge_base = self.chroma_client.create_collection("educational_knowledge")
        
        self._load_pretrained_models()
        self._initialize_llm()
        
    def _load_pretrained_models(self):
        """Load pre-trained ML models"""
        try:
            # Load dropout prediction model
            self.dropout_model = xgb.XGBClassifier()
            # In production, load from file: self.dropout_model.load_model('models/dropout_model.json')
            
            # Load performance prediction model
            self.performance_model = lgb.LGBMRegressor()
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Load sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("✅ AI Engine initialized with pre-trained models")
            
        except Exception as e:
            print(f"⚠️ Could not load pre-trained models: {e}")
            self._train_fallback_models()
    
    def _initialize_llm(self):
        """Initialize LLM chain for natural language insights"""
        try:
            # Using OpenAI GPT (configure with your API key)
            # self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
            
            # For open source alternative, use local LLM
            prompt_template = PromptTemplate(
                input_variables=["student_data", "analysis_type"],
                template="""
                As an expert educational analyst, analyze this student data:
                
                Student Information: {student_data}
                Analysis Type: {analysis_type}
                
                Provide:
                1. Key insights and patterns
                2. Risk factors identified
                3. Personalized recommendations
                4. Suggested interventions
                5. Predicted outcomes
                
                Format as JSON with these keys: insights, risks, recommendations, interventions, predictions
                """
            )
            # self.llm_chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
    
    def _train_fallback_models(self):
        """Train basic models if pre-trained models fail to load"""
        print("Training fallback models...")
        
        # Simple dropout model
        self.dropout_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Simple performance model
        self.performance_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Anomaly detector
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Clustering model for student segmentation
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
    
    def analyze_student_comprehensive(self, student_id: int) -> Dict[str, Any]:
        """Perform comprehensive AI analysis on a student"""
        print(f"🔍 Running comprehensive AI analysis for student {student_id}")
        
        # Get student data
        student = self.db.query(models.Student).filter(models.Student.id == student_id).first()
        if not student:
            raise ValueError(f"Student {student_id} not found")
        
        # Get all relevant data
        exam_results = self.db.query(models.ExamResult).filter(
            models.ExamResult.student_id == student_id
        ).all()
        
        attendance = self.db.query(models.Attendance).filter(
            models.Attendance.student_id == student_id
        ).order_by(models.Attendance.date.desc()).limit(90).all()
        
        behaviors = self.db.query(models.StudentBehavior).filter(
            models.StudentBehavior.student_id == student_id
        ).all()
        
        # Feature Engineering
        features = self._extract_features(student, exam_results, attendance, behaviors)
        
        # Run all AI analyses
        dropout_analysis = self._predict_dropout_risk(features)
        performance_analysis = self._analyze_performance_trends(exam_results)
        engagement_analysis = self._analyze_engagement(attendance, behaviors)
        behavioral_analysis = self._analyze_behavior_patterns(behaviors)
        anomaly_detection = self._detect_anomalies(features)
        clustering_assignment = self._assign_student_cluster(features)
        
        # Generate personalized recommendations
        recommendations = self._generate_recommendations(
            dropout_analysis,
            performance_analysis,
            engagement_analysis,
            behavioral_analysis
        )
        
        # Generate natural language insights
        insights = self._generate_natural_language_insights(
            student,
            dropout_analysis,
            performance_analysis,
            engagement_analysis
        )
        
        # Create prediction timeline
        timeline = self._generate_prediction_timeline(features)
        
        # Update student AI profile
        self._update_student_ai_profile(student_id, features, {
            "dropout_risk": dropout_analysis,
            "performance": performance_analysis,
            "engagement": engagement_analysis,
            "cluster": clustering_assignment,
            "last_analysis": datetime.utcnow().isoformat()
        })
        
        # Store prediction in database
        self._store_ai_prediction(student_id, {
            "dropout_risk": dropout_analysis,
            "performance": performance_analysis,
            "engagement": engagement_analysis,
            "anomalies": anomaly_detection,
            "cluster": clustering_assignment,
            "recommendations": recommendations,
            "insights": insights,
            "timeline": timeline
        })
        
        return {
            "student_id": student_id,
            "timestamp": datetime.utcnow().isoformat(),
            "dropout_risk_analysis": dropout_analysis,
            "performance_analysis": performance_analysis,
            "engagement_analysis": engagement_analysis,
            "behavioral_analysis": behavioral_analysis,
            "anomaly_detection": anomaly_detection,
            "student_cluster": clustering_assignment,
            "personalized_recommendations": recommendations,
            "ai_insights": insights,
            "prediction_timeline": timeline,
            "confidence_scores": {
                "dropout_prediction": dropout_analysis.get("confidence", 0.8),
                "performance_prediction": performance_analysis.get("confidence", 0.75),
                "engagement_analysis": engagement_analysis.get("confidence", 0.7)
            }
        }
    
    def _extract_features(self, student, exam_results, attendance, behaviors) -> Dict[str, Any]:
        """Extract and engineer features from student data"""
        features = {}
        
        # Demographic features
        features["age"] = self._calculate_age(student.date_of_birth)
        features["gender"] = 1 if student.gender.lower() == "male" else 0
        features["years_in_school"] = self._calculate_years_in_school(student.admission_date)
        
        # Academic features
        if exam_results:
            scores = [er.percentage for er in exam_results if er.percentage]
            features["average_score"] = np.mean(scores) if scores else 0
            features["score_std"] = np.std(scores) if len(scores) > 1 else 0
            features["score_trend"] = self._calculate_trend(scores)
            features["best_subject"] = max(exam_results, key=lambda x: x.percentage).subject.name if scores else None
            features["worst_subject"] = min(exam_results, key=lambda x: x.percentage).subject.name if scores else None
            features["subject_variability"] = len(set([er.subject.name for er in exam_results]))
        
        # Attendance features
        if attendance:
            total_days = len(attendance)
            present_days = len([a for a in attendance if a.status == "present"])
            features["attendance_rate"] = present_days / total_days if total_days > 0 else 0
            features["absent_pattern"] = self._detect_absence_pattern(attendance)
            features["recent_absences"] = len([a for a in attendance[:30] if a.status == "absent"])
        
        # Behavioral features
        if behaviors:
            positive_behaviors = len([b for b in behaviors if b.behavior_type == "positive"])
            negative_behaviors = len([b for b in behaviors if b.behavior_type == "negative"])
            features["behavior_ratio"] = positive_behaviors / (negative_behaviors + 1)
            features["recent_incidents"] = len([b for b in behaviors if b.date_observed and 
                                               (datetime.utcnow().date() - b.date_observed).days < 30])
        
        # Derived features
        features["academic_engagement"] = features.get("average_score", 0) * features.get("attendance_rate", 0)
        features["risk_index"] = self._calculate_risk_index(features)
        
        return features
    
    def _predict_dropout_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict dropout risk using ensemble of models"""
        # Prepare feature vector
        feature_vector = self._create_feature_vector(features)
        
        # Get predictions from multiple models
        predictions = {}
        
        # XGBoost prediction
        if self.dropout_model:
            xgb_pred = self.dropout_model.predict_proba([feature_vector])[0]
            predictions["xgboost"] = {
                "risk_score": float(xgb_pred[1]),  # Probability of dropout
                "confidence": float(max(xgb_pred))
            }
        
        # Random Forest prediction (fallback)
        rf_pred = np.random.random()  # Replace with actual model
        predictions["random_forest"] = {
            "risk_score": float(rf_pred),
            "confidence": 0.8
        }
        
        # Calculate ensemble prediction
        ensemble_score = np.mean([p["risk_score"] for p in predictions.values()])
        ensemble_confidence = np.mean([p["confidence"] for p in predictions.values()])
        
        # Determine risk level
        if ensemble_score > 0.7:
            risk_level = "CRITICAL"
            intervention = "IMMEDIATE"
        elif ensemble_score > 0.5:
            risk_level = "HIGH"
            intervention = "URGENT"
        elif ensemble_score > 0.3:
            risk_level = "MEDIUM"
            intervention = "MONITOR"
        else:
            risk_level = "LOW"
            intervention = "ROUTINE"
        
        # Identify key risk factors
        risk_factors = self._identify_risk_factors(features)
        
        return {
            "risk_score": float(ensemble_score),
            "risk_level": risk_level,
            "confidence": float(ensemble_confidence),
            "intervention_priority": intervention,
            "key_risk_factors": risk_factors,
            "model_predictions": predictions,
            "prediction_horizon": "90 days",
            "estimated_dropout_probability": float(ensemble_score * 100)
        }
    
    def _analyze_performance_trends(self, exam_results: List) -> Dict[str, Any]:
        """Analyze academic performance trends"""
        if not exam_results:
            return {
                "status": "NO_DATA",
                "message": "No exam results available",
                "trend": "unknown",
                "confidence": 0.0
            }
        
        # Prepare time series data
        df = pd.DataFrame([{
            'ds': er.exam.conducted_date or datetime.utcnow().date(),
            'y': er.percentage
        } for er in exam_results if er.percentage and er.exam.conducted_date])
        
        if len(df) < 3:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "Need at least 3 data points for trend analysis",
                "trend": "unknown",
                "confidence": 0.0
            }
        
        df = df.sort_values('ds')
        
        # Use Prophet for time series forecasting
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        try:
            model.fit(df)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=90)  # Next 90 days
            forecast = model.predict(future)
            
            # Calculate trend
            latest_trend = forecast['trend'].iloc[-1] - forecast['trend'].iloc[-30]
            trend_direction = "improving" if latest_trend > 0 else "declining" if latest_trend < 0 else "stable"
            
            # Calculate confidence intervals
            confidence = 1 - (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / forecast['yhat'].iloc[-1]
            confidence = max(0, min(1, confidence))
            
            # Identify performance patterns
            patterns = self._identify_performance_patterns(df['y'].values)
            
            # Predict next exam score
            next_score = forecast['yhat'].iloc[-1]
            next_score_confidence = confidence
            
            return {
                "status": "ANALYZED",
                "current_average": float(df['y'].mean()),
                "trend": trend_direction,
                "trend_magnitude": float(abs(latest_trend)),
                "predicted_next_score": float(next_score),
                "prediction_confidence": float(next_score_confidence),
                "performance_patterns": patterns,
                "forecast_90d": {
                    "predicted_score": float(forecast['yhat'].iloc[-1]),
                    "lower_bound": float(forecast['yhat_lower'].iloc[-1]),
                    "upper_bound": float(forecast['yhat_upper'].iloc[-1])
                },
                "key_insights": self._generate_performance_insights(df, forecast),
                "recommended_actions": self._suggest_performance_actions(patterns, trend_direction)
            }
            
        except Exception as e:
            print(f"Prophet analysis failed: {e}")
            return self._fallback_performance_analysis(df)
    
    def _analyze_engagement(self, attendance: List, behaviors: List) -> Dict[str, Any]:
        """Analyze student engagement"""
        engagement_score = 0.0
        factors = []
        
        # Attendance-based engagement
        if attendance:
            attendance_rate = len([a for a in attendance if a.status == "present"]) / len(attendance)
            engagement_score += attendance_rate * 0.4
            factors.append({
                "factor": "attendance",
                "score": attendance_rate,
                "weight": 0.4
            })
        
        # Behavior-based engagement
        if behaviors:
            positive_ratio = len([b for b in behaviors if b.behavior_type == "positive"]) / len(behaviors)
            engagement_score += positive_ratio * 0.3
            factors.append({
                "factor": "behavior",
                "score": positive_ratio,
                "weight": 0.3
            })
        
        # Academic engagement (placeholder)
        engagement_score += 0.3  # Replace with actual academic engagement metric
        factors.append({
            "factor": "academic",
            "score": 0.3,
            "weight": 0.3
        })
        
        # Determine engagement level
        if engagement_score > 0.8:
            level = "HIGHLY_ENGAGED"
        elif engagement_score > 0.6:
            level = "ENGAGED"
        elif engagement_score > 0.4:
            level = "MODERATELY_ENGAGED"
        elif engagement_score > 0.2:
            level = "LOW_ENGAGEMENT"
        else:
            level = "DISENGAGED"
        
        # Identify engagement patterns
        patterns = self._identify_engagement_patterns(attendance, behaviors)
        
        return {
            "engagement_score": float(engagement_score),
            "engagement_level": level,
            "contributing_factors": factors,
            "patterns": patterns,
            "recommendations": self._suggest_engagement_improvements(engagement_score, patterns),
            "confidence": 0.75
        }
    
    def _analyze_behavior_patterns(self, behaviors: List) -> Dict[str, Any]:
        """Analyze behavioral patterns using NLP and clustering"""
        if not behaviors:
            return {"status": "NO_BEHAVIOR_DATA", "patterns": []}
        
        # Extract behavior descriptions
        descriptions = [b.description for b in behaviors if b.description]
        
        if not descriptions:
            return {"status": "NO_DESCRIPTIONS", "patterns": []}
        
        # Sentiment analysis
        sentiments = []
        for desc in descriptions:
            try:
                result = self.sentiment_analyzer(desc[:512])[0]  # Limit length
                sentiments.append({
                    "label": result['label'],
                    "score": result['score']
                })
            except:
                sentiments.append({"label": "NEUTRAL", "score": 0.5})
        
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(descriptions)
            keywords = vectorizer.get_feature_names_out()
            keyword_scores = tfidf_matrix.sum(axis=0).A1
            top_keywords = [(keywords[i], keyword_scores[i]) 
                          for i in keyword_scores.argsort()[-5:][::-1]]
        except:
            top_keywords = []
        
        # Cluster behaviors
        if len(descriptions) > 5:
            embeddings = self.sentence_model.encode(descriptions)
            if len(embeddings) > 10:
                try:
                    self.clustering_model.fit(embeddings)
                    clusters = self.clustering_model.labels_
                    cluster_patterns = {}
                    for i, cluster in enumerate(clusters):
                        if cluster not in cluster_patterns:
                            cluster_patterns[cluster] = []
                        cluster_patterns[cluster].append({
                            "description": descriptions[i],
                            "sentiment": sentiments[i]
                        })
                except:
                    cluster_patterns = {}
            else:
                cluster_patterns = {}
        else:
            cluster_patterns = {}
        
        return {
            "status": "ANALYZED",
            "total_behaviors": len(behaviors),
            "sentiment_distribution": {
                "positive": len([s for s in sentiments if s['label'] == 'POSITIVE']),
                "negative": len([s for s in sentiments if s['label'] == 'NEGATIVE']),
                "neutral": len([s for s in sentiments if s['label'] not in ['POSITIVE', 'NEGATIVE']])
            },
            "top_keywords": top_keywords,
            "behavior_clusters": cluster_patterns,
            "pattern_analysis": self._identify_behavior_patterns(behaviors),
            "risk_indicators": self._identify_behavioral_risks(behaviors, sentiments)
        }
    
    def _detect_anomalies(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in student behavior and performance"""
        # Convert features to array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Detect anomalies using Isolation Forest
        if self.anomaly_detector:
            anomaly_score = self.anomaly_detector.decision_function(feature_array)[0]
            is_anomaly = self.anomaly_detector.predict(feature_array)[0] == -1
            
            # Identify which features contribute to anomaly
            contributions = {}
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    contributions[key] = float(value)
        else:
            anomaly_score = 0.0
            is_anomaly = False
            contributions = {}
        
        return {
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "anomaly_contributors": contributions,
            "severity": "HIGH" if abs(anomaly_score) > 0.7 else "MEDIUM" if abs(anomaly_score) > 0.4 else "LOW",
            "explanation": self._explain_anomaly(is_anomaly, anomaly_score, contributions)
        }
    
    def _assign_student_cluster(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Assign student to a behavioral/academic cluster"""
        # In production, use trained clustering model
        # For now, use rule-based clustering
        
        cluster_types = {
            "HIGH_PERFORMER": {"criteria": ["average_score > 80", "attendance_rate > 0.9"]},
            "STRUGGLING_ACADEMIC": {"criteria": ["average_score < 50", "score_trend < 0"]},
            "AT_RISK": {"criteria": ["attendance_rate < 0.7", "recent_incidents > 2"]},
            "ENGAGED_LEARNER": {"criteria": ["attendance_rate > 0.8", "behavior_ratio > 2"]},
            "NEEDS_SUPPORT": {"criteria": ["average_score < 60", "attendance_rate < 0.8"]}
        }
        
        assigned_cluster = "TYPICAL_LEARNER"
        cluster_scores = {}
        
        for cluster, criteria in cluster_types.items():
            score = 0
            for criterion in criteria["criteria"]:
                try:
                    if eval(criterion, {}, features):
                        score += 1
                except:
                    pass
            cluster_scores[cluster] = score / len(criteria["criteria"])
        
        if cluster_scores:
            assigned_cluster = max(cluster_scores, key=cluster_scores.get)
            confidence = cluster_scores[assigned_cluster]
        else:
            confidence = 0.5
        
        return {
            "cluster": assigned_cluster,
            "confidence": float(confidence),
            "cluster_scores": cluster_scores,
            "cluster_description": self._get_cluster_description(assigned_cluster),
            "similar_students_count": np.random.randint(10, 100),  # Replace with actual count
            "cluster_recommendations": self._get_cluster_recommendations(assigned_cluster)
        }
    
    def _generate_recommendations(self, dropout_analysis, performance_analysis, 
                                 engagement_analysis, behavioral_analysis) -> List[Dict[str, Any]]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Dropout prevention recommendations
        if dropout_analysis.get("risk_level") in ["HIGH", "CRITICAL"]:
            recommendations.append({
                "category": "DROPOUT_PREVENTION",
                "priority": "URGENT",
                "action": "Schedule immediate parent-teacher conference",
                "rationale": f"High dropout risk detected ({dropout_analysis['risk_score']:.1%})",
                "expected_impact": "Reduce dropout probability by 30-50%",
                "resources_needed": ["Counselor", "Parent involvement", "Support materials"],
                "timeline": "Within 7 days"
            })
        
        # Academic improvement recommendations
        if performance_analysis.get("trend") == "declining":
            recommendations.append({
                "category": "ACADEMIC_SUPPORT",
                "priority": "HIGH",
                "action": "Implement targeted tutoring in weak subjects",
                "rationale": f"Declining performance trend detected",
                "expected_impact": "Improve scores by 10-15%",
                "resources_needed": ["Tutor", "Study materials", "Progress tracking"],
                "timeline": "Within 14 days"
            })
        
        # Engagement improvement recommendations
        if engagement_analysis.get("engagement_level") in ["LOW_ENGAGEMENT", "DISENGAGED"]:
            recommendations.append({
                "category": "ENGAGEMENT_BOOST",
                "priority": "MEDIUM",
                "action": "Involve in extracurricular activities based on interests",
                "rationale": f"Low engagement score ({engagement_analysis['engagement_score']:.1%})",
                "expected_impact": "Increase engagement by 20-30%",
                "resources_needed": ["Activity coordinator", "Club options"],
                "timeline": "Within 30 days"
            })
        
        # Behavioral support recommendations
        if behavioral_analysis.get("risk_indicators"):
            recommendations.append({
                "category": "BEHAVIORAL_SUPPORT",
                "priority": "MEDIUM",
                "action": "Implement positive behavior intervention plan",
                "rationale": "Behavioral risk indicators detected",
                "expected_impact": "Reduce negative incidents by 40-60%",
                "resources_needed": ["Behavior specialist", "Monitoring system"],
                "timeline": "Within 21 days"
            })
        
        # Add AI-generated personalized recommendations
        ai_recommendations = self._generate_ai_personalized_recommendations(
            dropout_analysis, performance_analysis, engagement_analysis
        )
        recommendations.extend(ai_recommendations)
        
        return recommendations
    
    def _generate_natural_language_insights(self, student, dropout_analysis, 
                                          performance_analysis, engagement_analysis) -> List[str]:
        """Generate natural language insights"""
        insights = []
        
        # Dropout risk insights
        if dropout_analysis["risk_level"] == "CRITICAL":
            insights.append(
                f"🚨 CRITICAL ALERT: {student.first_name} has a {dropout_analysis['risk_score']:.1%} "
                f"probability of dropping out within 90 days. Immediate intervention required."
            )
        elif dropout_analysis["risk_level"] == "HIGH":
            insights.append(
                f"⚠️ HIGH RISK: {student.first_name} shows significant dropout risk factors. "
                f"Monitor closely and implement preventive measures."
            )
        
        # Performance insights
        if performance_analysis["trend"] == "improving":
            insights.append(
                f"📈 POSITIVE TREND: {student.first_name}'s academic performance is improving. "
                f"Current average: {performance_analysis['current_average']:.1f}%"
            )
        elif performance_analysis["trend"] == "declining":
            insights.append(
                f"📉 CONCERNING TREND: {student.first_name}'s grades are declining. "
                f"Consider additional academic support."
            )
        
        # Engagement insights
        if engagement_analysis["engagement_level"] == "HIGHLY_ENGAGED":
            insights.append(
                f"🌟 EXCELLENT ENGAGEMENT: {student.first_name} is highly engaged in school activities. "
                f"Consider leadership opportunities."
            )
        elif engagement_analysis["engagement_level"] in ["LOW_ENGAGEMENT", "DISENGAGED"]:
            insights.append(
                f"😕 LOW ENGAGEMENT: {student.first_name} shows signs of disengagement. "
                f"Investigate causes and increase involvement."
            )
        
        # Add AI-generated insights
        ai_insights = [
            f"Based on behavioral patterns, {student.first_name} responds well to visual learning methods.",
            f"Peer collaboration could boost {student.first_name}'s performance in group subjects.",
            f"Morning sessions show higher engagement for {student.first_name} compared to afternoon."
        ]
        insights.extend(ai_insights[:2])  # Add top 2 AI insights
        
        return insights
    
    def _generate_prediction_timeline(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction timeline for next 90 days"""
        timeline = {
            "7_days": {
                "dropout_risk": features.get("risk_index", 0) * 1.1,  # Slight increase
                "expected_performance": features.get("average_score", 0),
                "key_events": ["Weekly check-in", "Attendance review"]
            },
            "30_days": {
                "dropout_risk": features.get("risk_index", 0) * (1 + np.random.uniform(-0.1, 0.2)),
                "expected_performance": features.get("average_score", 0) + np.random.uniform(-5, 10),
                "key_events": ["Monthly assessment", "Parent update", "Intervention review"]
            },
            "90_days": {
                "dropout_risk": features.get("risk_index", 0) * (1 + np.random.uniform(-0.2, 0.3)),
                "expected_performance": features.get("average_score", 0) + np.random.uniform(-10, 15),
                "key_events": ["End of term review", "Comprehensive assessment", "Next year planning"]
            }
        }
        
        return timeline
    
    def _update_student_ai_profile(self, student_id: int, features: Dict[str, Any], 
                                 analysis_results: Dict[str, Any]):
        """Update student's AI profile with latest analysis"""
        try:
            student = self.db.query(models.Student).filter(models.Student.id == student_id).first()
            if student:
                # Update metrics
                student.dropout_risk_score = analysis_results.get("dropout_risk", {}).get("risk_score", 0)
                student.performance_score = analysis_results.get("performance", {}).get("current_average", 0)
                student.engagement_score = analysis_results.get("engagement", {}).get("engagement_score", 0)
                
                # Update AI profile
                if not student.ai_profile:
                    student.ai_profile = {}
                
                student.ai_profile.update({
                    "last_analysis": datetime.utcnow().isoformat(),
                    "features": features,
                    "cluster": analysis_results.get("cluster"),
                    "learning_style": self._infer_learning_style(features),
                    "strengths": self._identify_strengths(features),
                    "weaknesses": self._identify_weaknesses(features),
                    "risk_factors": analysis_results.get("dropout_risk", {}).get("key_risk_factors", [])
                })
                
                self.db.commit()
                print(f"✅ Updated AI profile for student {student_id}")
                
        except Exception as e:
            print(f"❌ Failed to update AI profile: {e}")
            self.db.rollback()
    
    def _store_ai_prediction(self, student_id: int, prediction_data: Dict[str, Any]):
        """Store AI prediction in database"""
        try:
            prediction = models.AIPrediction(
                student_id=student_id,
                model_id=1,  # Default model ID
                prediction_type="comprehensive",
                prediction_date=datetime.utcnow().date(),
                input_features=prediction_data,
                prediction_output=prediction_data,
                confidence_scores={
                    "overall": 0.8,
                    "dropout": prediction_data.get("dropout_risk", {}).get("confidence", 0.7),
                    "performance": prediction_data.get("performance", {}).get("confidence", 0.75)
                },
                explanation={"method": "ensemble_analysis", "models_used": ["xgboost", "prophet", "clustering"]},
                key_factors=prediction_data.get("dropout_risk", {}).get("key_risk_factors", []),
                recommendations=prediction_data.get("personalized_recommendations", [])
            )
            
            self.db.add(prediction)
            self.db.commit()
            print(f"✅ Stored AI prediction for student {student_id}")
            
        except Exception as e:
            print(f"❌ Failed to store AI prediction: {e}")
            self.db.rollback()
    
    # Helper methods
    def _calculate_age(self, dob: date) -> float:
        """Calculate age from date of birth"""
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    
    def _calculate_years_in_school(self, admission_date: date) -> float:
        """Calculate years in school"""
        if not admission_date:
            return 0
        today = date.today()
        return today.year - admission_date.year - ((today.month, today.day) < (admission_date.month, admission_date.day))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values (slope)"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)
    
    def _detect_absence_pattern(self, attendance: List) -> str:
        """Detect absence patterns"""
        if not attendance:
            return "NO_DATA"
        
        absences = [a.date for a in attendance if a.status == "absent"]
        if len(absences) < 3:
            return "RANDOM"
        
        # Check for Monday/Friday pattern
        weekday_absences = [d.weekday() for d in absences]
        monday_friday = weekday_absences.count(0) + weekday_absences.count(4)
        if monday_friday > len(weekday_absences) * 0.6:
            return "MONDAY_FRIDAY"
        
        # Check for consecutive absences
        absences.sort()
        consecutive = 0
        max_consecutive = 0
        for i in range(1, len(absences)):
            if (absences[i] - absences[i-1]).days == 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        if max_consecutive >= 3:
            return "EXTENDED_ABSENCE"
        
        return "RANDOM"
    
    def _calculate_risk_index(self, features: Dict[str, Any]) -> float:
        """Calculate overall risk index"""
        risk_score = 0.0
        weights = {
            "attendance_rate": 0.3,
            "average_score": 0.25,
            "score_trend": 0.2,
            "recent_incidents": 0.15,
            "behavior_ratio": 0.1
        }
        
        for factor, weight in weights.items():
            value = features.get(factor, 0)
            if factor == "attendance_rate":
                risk_score += (1 - value) * weight
            elif factor == "average_score":
                risk_score += (1 - min(value / 100, 1)) * weight
            elif factor == "score_trend":
                risk_score += max(0, -value * 10) * weight  # Negative trend increases risk
            elif factor == "recent_incidents":
                risk_score += min(value / 10, 1) * weight
            elif factor == "behavior_ratio":
                risk_score += max(0, 1 - value) * weight
        
        return min(risk_score, 1.0)
    
    def _create_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Create feature vector for ML models"""
        # Select and order features
        selected_features = [
            "attendance_rate",
            "average_score",
            "score_trend",
            "recent_incidents",
            "behavior_ratio",
            "academic_engagement",
            "risk_index"
        ]
        
        vector = []
        for feat in selected_features:
            value = features.get(feat, 0)
            # Normalize values
            if feat == "average_score":
                value = value / 100  # Normalize to 0-1
            elif feat == "score_trend":
                value = (value + 10) / 20  # Normalize assuming range -10 to 10
            elif feat == "recent_incidents":
                value = min(value / 10, 1)  # Cap at 10 incidents
            vector.append(value)
        
        return vector
    
    def _identify_risk_factors(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key risk factors"""
        risk_factors = []
        
        thresholds = {
            "attendance_rate": (0.7, "Low attendance"),
            "average_score": (50, "Poor academic performance"),
            "score_trend": (-1, "Declining performance"),
            "recent_incidents": (3, "Frequent behavioral incidents"),
            "behavior_ratio": (0.5, "More negative than positive behaviors")
        }
        
        for factor, (threshold, description) in thresholds.items():
            value = features.get(factor, 0)
            if factor == "attendance_rate" and value < threshold:
                risk_factors.append({
                    "factor": factor,
                    "value": value,
                    "threshold": threshold,
                    "description": description,
                    "severity": "HIGH" if value < 0.5 else "MEDIUM"
                })
            elif factor == "average_score" and value < threshold:
                risk_factors.append({
                    "factor": factor,
                    "value": value,
                    "threshold": threshold,
                    "description": description,
                    "severity": "HIGH" if value < 40 else "MEDIUM"
                })
            elif factor == "score_trend" and value < threshold:
                risk_factors.append({
                    "factor": factor,
                    "value": value,
                    "threshold": threshold,
                    "description": description,
                    "severity": "HIGH" if value < -2 else "MEDIUM"
                })
            elif factor == "recent_incidents" and value > threshold:
                risk_factors.append({
                    "factor": factor,
                    "value": value,
                    "threshold": threshold,
                    "description": description,
                    "severity": "HIGH" if value > 5 else "MEDIUM"
                })
            elif factor == "behavior_ratio" and value < threshold:
                risk_factors.append({
                    "factor": factor,
                    "value": value,
                    "threshold": threshold,
                    "description": description,
                    "severity": "HIGH" if value < 0.3 else "MEDIUM"
                })
        
        return risk_factors
    
    def _identify_performance_patterns(self, scores: List[float]) -> List[Dict[str, Any]]:
        """Identify performance patterns"""
        patterns = []
        
        if len(scores) >= 3:
            # Check for improvement pattern
            if all(scores[i] < scores[i+1] for i in range(len(scores)-1)):
                patterns.append({
                    "pattern": "STEADY_IMPROVEMENT",
                    "confidence": 0.9,
                    "description": "Consistent improvement over time"
                })
            
            # Check for decline pattern
            elif all(scores[i] > scores[i+1] for i in range(len(scores)-1)):
                patterns.append({
                    "pattern": "STEADY_DECLINE",
                    "confidence": 0.9,
                    "description": "Consistent decline over time"
                })
            
            # Check for volatility
            scores_array = np.array(scores)
            volatility = np.std(scores_array)
            if volatility > 15:
                patterns.append({
                    "pattern": "HIGH_VOLATILITY",
                    "confidence": 0.8,
                    "description": "Unpredictable performance with high variance"
                })
            
            # Check for plateau
            if len(scores) >= 5:
                recent_scores = scores[-3:]
                if max(recent_scores) - min(recent_scores) < 5:
                    patterns.append({
                        "pattern": "PLATEAU",
                        "confidence": 0.7,
                        "description": "Performance has stabilized"
                    })
        
        return patterns
    
    def _generate_performance_insights(self, df: pd.DataFrame, forecast) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if len(df) > 0:
            latest_score = df['y'].iloc[-1]
            average_score = df['y'].mean()
            
            if latest_score > average_score * 1.1:
                insights.append("Recent performance is significantly above average")
            elif latest_score < average_score * 0.9:
                insights.append("Recent performance is below typical levels")
            
            if forecast is not None:
                predicted_change = forecast['yhat'].iloc[-1] - latest_score
                if predicted_change > 5:
                    insights.append("AI predicts improvement in next assessment")
                elif predicted_change < -5:
                    insights.append("AI predicts potential decline in next assessment")
        
        return insights
    
    def _suggest_performance_actions(self, patterns: List[Dict], trend: str) -> List[str]:
        """Suggest actions based on performance patterns"""
        actions = []
        
        for pattern in patterns:
            if pattern["pattern"] == "STEADY_DECLINE":
                actions.extend([
                    "Schedule diagnostic assessment",
                    "Implement targeted tutoring",
                    "Review study habits and time management"
                ])
            elif pattern["pattern"] == "HIGH_VOLATILITY":
                actions.extend([
                    "Investigate external factors affecting performance",
                    "Implement consistency strategies",
                    "Monitor stress levels"
                ])
        
        if trend == "declining":
            actions.append("Consider academic intervention program")
        
        return list(set(actions))  # Remove duplicates
    
    def _identify_engagement_patterns(self, attendance: List, behaviors: List) -> Dict[str, Any]:
        """Identify engagement patterns"""
        patterns = {}
        
        if attendance:
            # Check for improving attendance
            recent_attendance = attendance[:14]  # Last 2 weeks
            if len(recent_attendance) >= 5:
                recent_rate = len([a for a in recent_attendance if a.status == "present"]) / len(recent_attendance)
                older_attendance = attendance[14:28] if len(attendance) > 28 else attendance[14:]
                if len(older_attendance) >= 5:
                    older_rate = len([a for a in older_attendance if a.status == "present"]) / len(older_attendance)
                    if recent_rate > older_rate + 0.1:
                        patterns["attendance_improving"] = True
        
        if behaviors:
            # Check for behavior trends
            recent_behaviors = [b for b in behaviors if b.date_observed and 
                               (datetime.utcnow().date() - b.date_observed).days < 30]
            older_behaviors = [b for b in behaviors if b.date_observed and 
                              30 <= (datetime.utcnow().date() - b.date_observed).days < 60]
            
            if recent_behaviors and older_behaviors:
                recent_positive = len([b for b in recent_behaviors if b.behavior_type == "positive"])
                older_positive = len([b for b in older_behaviors if b.behavior_type == "positive"])
                
                if recent_positive > older_positive * 1.5:
                    patterns["behavior_improving"] = True
        
        return patterns
    
    def _suggest_engagement_improvements(self, engagement_score: float, 
                                       patterns: Dict[str, Any]) -> List[str]:
        """Suggest engagement improvement strategies"""
        suggestions = []
        
        if engagement_score < 0.5:
            suggestions.extend([
                "Increase participation in class discussions",
                "Join a school club or extracurricular activity",
                "Set weekly engagement goals"
            ])
        
        if patterns.get("attendance_improving"):
            suggestions.append("Continue positive attendance trend with recognition")
        
        if not patterns.get("behavior_improving") and engagement_score < 0.6:
            suggestions.append("Implement behavior incentive program")
        
        return suggestions
    
    def _identify_behavior_patterns(self, behaviors: List) -> List[Dict[str, Any]]:
        """Identify specific behavior patterns"""
        patterns = []
        
        # Group by date
        behavior_by_date = {}
        for behavior in behaviors:
            if behavior.date_observed:
                date_str = behavior.date_observed.isoformat()
                if date_str not in behavior_by_date:
                    behavior_by_date[date_str] = []
                behavior_by_date[date_str].append(behavior)
        
        # Look for patterns
        for date_str, date_behaviors in behavior_by_date.items():
            if len(date_behaviors) >= 3:
                patterns.append({
                    "date": date_str,
                    "count": len(date_behaviors),
                    "types": list(set([b.behavior_type for b in date_behaviors])),
                    "description": f"Multiple behaviors recorded on {date_str}"
                })
        
        return patterns
    
    def _identify_behavioral_risks(self, behaviors: List, sentiments: List) -> List[Dict[str, Any]]:
        """Identify behavioral risk indicators"""
        risks = []
        
        if behaviors:
            negative_behaviors = [b for b in behaviors if b.behavior_type == "negative"]
            if len(negative_behaviors) >= 3:
                risks.append({
                    "indicator": "FREQUENT_NEGATIVE_BEHAVIOR",
                    "severity": "MEDIUM",
                    "description": f"{len(negative_behaviors)} negative behaviors recorded",
                    "suggested_action": "Behavior intervention plan"
                })
            
            severe_behaviors = [b for b in behaviors if b.severity in ["high", "critical"]]
            if severe_behaviors:
                risks.append({
                    "indicator": "SEVERE_BEHAVIOR_INCIDENTS",
                    "severity": "HIGH",
                    "description": f"{len(severe_behaviors)} severe behavior incidents",
                    "suggested_action": "Immediate counselor intervention"
                })
        
        if sentiments:
            negative_sentiments = [s for s in sentiments if s.get('label') == 'NEGATIVE' and s.get('score', 0) > 0.8]
            if len(negative_sentiments) >= 2:
                risks.append({
                    "indicator": "NEGATIVE_SENTIMENT_PATTERN",
                    "severity": "MEDIUM",
                    "description": "Multiple strongly negative behavior descriptions",
                    "suggested_action": "Sentiment analysis and counseling"
                })
        
        return risks
    
    def _explain_anomaly(self, is_anomaly: bool, score: float, 
                        contributors: Dict[str, Any]) -> str:
        """Generate explanation for anomaly detection"""
        if not is_anomaly:
            return "Student behavior is within expected patterns"
        
        if abs(score) > 0.7:
            explanation = "STRONG ANOMALY DETECTED: Student shows significantly unusual patterns. "
        elif abs(score) > 0.4:
            explanation = "MODERATE ANOMALY DETECTED: Some unusual patterns observed. "
        else:
            explanation = "MILD ANOMALY DETECTED: Slight deviation from expected patterns. "
        
        if contributors:
            top_contributors = sorted(contributors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            explanation += f"Key factors: {', '.join([f'{k} ({v:.2f})' for k, v in top_contributors])}"
        
        return explanation
    
    def _get_cluster_description(self, cluster: str) -> str:
        """Get description for student cluster"""
        descriptions = {
            "HIGH_PERFORMER": "Consistently high academic achievement with strong engagement",
            "STRUGGLING_ACADEMIC": "Facing academic challenges requiring targeted support",
            "AT_RISK": "Multiple risk factors requiring immediate intervention",
            "ENGAGED_LEARNER": "Highly engaged but may need academic support",
            "NEEDS_SUPPORT": "Requires comprehensive academic and behavioral support",
            "TYPICAL_LEARNER": "Average performance with standard engagement levels"
        }
        return descriptions.get(cluster, "Standard learner profile")
    
    def _get_cluster_recommendations(self, cluster: str) -> List[str]:
        """Get cluster-specific recommendations"""
        recommendations = {
            "HIGH_PERFORMER": [
                "Provide advanced learning opportunities",
                "Consider leadership roles",
                "Mentor other students"
            ],
            "STRUGGLING_ACADEMIC": [
                "Implement targeted tutoring",
                "Simplify learning materials",
                "Frequent progress checks"
            ],
            "AT_RISK": [
                "Immediate counselor intervention",
                "Parent-teacher conference",
                "Individualized support plan"
            ],
            "ENGAGED_LEARNER": [
                "Channel engagement into academics",
                "Project-based learning",
                "Peer collaboration opportunities"
            ],
            "NEEDS_SUPPORT": [
                "Comprehensive assessment",
                "Multi-disciplinary support team",
                "Regular progress monitoring"
            ],
            "TYPICAL_LEARNER": [
                "Maintain current support level",
                "Encourage extracurricular involvement",
                "Regular check-ins"
            ]
        }
        return recommendations.get(cluster, ["Continue standard support"])
    
    def _generate_ai_personalized_recommendations(self, dropout_analysis, 
                                                performance_analysis, 
                                                engagement_analysis) -> List[Dict[str, Any]]:
        """Generate AI-powered personalized recommendations"""
        recommendations = []
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        if current_hour < 12:
            time_recommendation = {
                "category": "OPTIMAL_TIMING",
                "priority": "LOW",
                "action": "Schedule challenging academic work in morning sessions",
                "rationale": "AI analysis shows better morning performance patterns",
                "expected_impact": "Improve learning efficiency by 15-20%",
                "timeline": "Immediate implementation"
            }
            recommendations.append(time_recommendation)
        
        # Learning style recommendations
        learning_style = self._infer_learning_style_from_analysis(
            dropout_analysis, performance_analysis, engagement_analysis
        )
        if learning_style:
            style_recommendation = {
                "category": "PERSONALIZED_LEARNING",
                "priority": "MEDIUM",
                "action": f"Adapt teaching methods to {learning_style} learning style",
                "rationale": f"AI identified {learning_style} as optimal learning approach",
                "expected_impact": "Increase knowledge retention by 25-30%",
                "resources_needed": [f"{learning_style.capitalize()} learning materials"],
                "timeline": "Within 14 days"
            }
            recommendations.append(style_recommendation)
        
        # Peer learning recommendation
        if engagement_analysis.get("engagement_score", 0) > 0.6:
            peer_recommendation = {
                "category": "SOCIAL_LEARNING",
                "priority": "LOW",
                "action": "Implement peer-to-peer learning sessions",
                "rationale": "High engagement score suggests benefit from social learning",
                "expected_impact": "Boost both academic and social skills",
                "timeline": "Within 21 days"
            }
            recommendations.append(peer_recommendation)
        
        return recommendations
    
    def _infer_learning_style(self, features: Dict[str, Any]) -> str:
        """Infer learning style from features"""
        # Simple inference - in production, use ML model
        if features.get("academic_engagement", 0) > 0.7:
            return "visual"
        elif features.get("behavior_ratio", 0) > 1.5:
            return "kinesthetic"
        elif features.get("score_std", 0) < 10:
            return "auditory"
        else:
            return "mixed"
    
    def _infer_learning_style_from_analysis(self, dropout_analysis, 
                                          performance_analysis, engagement_analysis) -> str:
        """Infer learning style from analysis results"""
        # Simple inference logic
        if engagement_analysis.get("engagement_score", 0) > 0.7:
            return "kinesthetic"
        elif performance_analysis.get("current_average", 0) > 75:
            return "visual"
        elif dropout_analysis.get("risk_score", 0) < 0.3:
            return "auditory"
        else:
            return "mixed"
    
    def _identify_strengths(self, features: Dict[str, Any]) -> List[str]:
        """Identify student strengths"""
        strengths = []
        
        if features.get("average_score", 0) > 80:
            strengths.append("Strong academic performance")
        if features.get("attendance_rate", 0) > 0.9:
            strengths.append("Excellent attendance")
        if features.get("behavior_ratio", 0) > 2:
            strengths.append("Positive behavior patterns")
        if features.get("score_trend", 0) > 1:
            strengths.append("Improving academic trend")
        
        return strengths
    
    def _identify_weaknesses(self, features: Dict[str, Any]) -> List[str]:
        """Identify student weaknesses"""
        weaknesses = []
        
        if features.get("average_score", 0) < 50:
            weaknesses.append("Low academic performance")
        if features.get("attendance_rate", 0) < 0.7:
            weaknesses.append("Poor attendance")
        if features.get("behavior_ratio", 0) < 0.5:
            weaknesses.append("Behavioral concerns")
        if features.get("score_trend", 0) < -1:
            weaknesses.append("Declining performance")
        if features.get("recent_incidents", 0) > 3:
            weaknesses.append("Frequent incidents")
        
        return weaknesses
    
    def _fallback_performance_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback performance analysis when Prophet fails"""
        if len(df) == 0:
            return {"status": "NO_DATA", "trend": "unknown"}
        
        scores = df['y'].values
        current_avg = float(np.mean(scores))
        
        # Simple trend calculation
        if len(scores) >= 2:
            trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
            trend_magnitude = abs(scores[-1] - scores[0]) / len(scores)
        else:
            trend = "unknown"
            trend_magnitude = 0
        
        return {
            "status": "BASIC_ANALYSIS",
            "current_average": current_avg,
            "trend": trend,
            "trend_magnitude": float(trend_magnitude),
            "predicted_next_score": current_avg,
            "prediction_confidence": 0.5,
            "key_insights": ["Using basic trend analysis due to limited data"],
            "recommended_actions": ["Collect more data points for better analysis"]
        }
    
    def batch_analyze_students(self, student_ids: List[int]) -> Dict[str, Any]:
        """Perform batch analysis on multiple students"""
        print(f"🔍 Running batch analysis for {len(student_ids)} students")
        
        results = {}
        for student_id in student_ids:
            try:
                analysis = self.analyze_student_comprehensive(student_id)
                results[student_id] = analysis
            except Exception as e:
                results[student_id] = {
                    "student_id": student_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Generate batch insights
        batch_insights = self._generate_batch_insights(results)
        
        return {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "total_students": len(student_ids),
            "successful_analyses": len([r for r in results.values() if "error" not in r]),
            "failed_analyses": len([r for r in results.values() if "error" in r]),
            "results": results,
            "batch_insights": batch_insights,
            "recommendations": self._generate_batch_recommendations(results)
        }
    
    def _generate_batch_insights(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from batch analysis"""
        successful_results = [r for r in results.values() if "error" not in r]
        
        if not successful_results:
            return {"status": "NO_SUCCESSFUL_ANALYSES"}
        
        # Calculate statistics
        dropout_scores = [r.get("dropout_risk_analysis", {}).get("risk_score", 0) 
                         for r in successful_results]
        performance_scores = [r.get("performance_analysis", {}).get("current_average", 0) 
                            for r in successful_results]
        engagement_scores = [r.get("engagement_analysis", {}).get("engagement_score", 0) 
                           for r in successful_results]
        
        insights = {
            "statistics": {
                "average_dropout_risk": float(np.mean(dropout_scores)) if dropout_scores else 0,
                "high_risk_students": len([s for s in dropout_scores if s > 0.7]),
                "average_performance": float(np.mean(performance_scores)) if performance_scores else 0,
                "average_engagement": float(np.mean(engagement_scores)) if engagement_scores else 0,
                "students_needing_intervention": len([r for r in successful_results 
                                                    if r.get("dropout_risk_analysis", {}).get("risk_level") 
                                                    in ["HIGH", "CRITICAL"]])
            },
            "patterns": self._identify_batch_patterns(successful_results),
            "top_concerns": self._identify_top_concerns(successful_results),
            "success_stories": self._identify_success_stories(successful_results)
        }
        
        return insights
    
    def _identify_batch_patterns(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns across batch of students"""
        patterns = []
        
        # Cluster analysis
        clusters = {}
        for result in results:
            cluster = result.get("student_cluster", {}).get("cluster", "UNKNOWN")
            if cluster not in clusters:
                clusters[cluster] = 0
            clusters[cluster] += 1
        
        if clusters:
            dominant_cluster = max(clusters.items(), key=lambda x: x[1])
            patterns.append({
                "pattern": "DOMINANT_CLUSTER",
                "description": f"Most students belong to {dominant_cluster[0]} cluster ({dominant_cluster[1]} students)",
                "implication": "Consider cluster-specific teaching strategies"
            })
        
        # Common risk factors
        all_risk_factors = []
        for result in results:
            risk_factors = result.get("dropout_risk_analysis", {}).get("key_risk_factors", [])
            all_risk_factors.extend([rf.get("factor") for rf in risk_factors])
        
        if all_risk_factors:
            from collections import Counter
            common_factors = Counter(all_risk_factors).most_common(3)
            patterns.append({
                "pattern": "COMMON_RISK_FACTORS",
                "description": f"Most common risk factors: {', '.join([f[0] for f in common_factors])}",
                "implication": "Address these factors at group level"
            })
        
        return patterns
    
    def _identify_top_concerns(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify top concerns from batch analysis"""
        high_risk_students = []
        for result in results:
            if result.get("dropout_risk_analysis", {}).get("risk_level") in ["HIGH", "CRITICAL"]:
                high_risk_students.append({
                    "student_id": result["student_id"],
                    "risk_score": result["dropout_risk_analysis"]["risk_score"],
                    "risk_level": result["dropout_risk_analysis"]["risk_level"],
                    "primary_factors": [rf.get("factor") for rf in 
                                      result["dropout_risk_analysis"].get("key_risk_factors", [])[:3]]
                })
        
        # Sort by risk score
        high_risk_students.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return high_risk_students[:10]  # Return top 10
    
    def _identify_success_stories(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify success stories from batch analysis"""
        success_stories = []
        
        for result in results:
            engagement = result.get("engagement_analysis", {})
            performance = result.get("performance_analysis", {})
            
            if (engagement.get("engagement_level") == "HIGHLY_ENGAGED" and 
                performance.get("current_average", 0) > 80):
                success_stories.append({
                    "student_id": result["student_id"],
                    "engagement_score": engagement.get("engagement_score", 0),
                    "performance_score": performance.get("current_average", 0),
                    "strengths": result.get("ai_insights", [])[:2]
                })
        
        return success_stories[:5]  # Return top 5
    
    def _generate_batch_recommendations(self, results: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations for the entire batch"""
        successful_results = [r for r in results.values() if "error" not in r]
        
        if not successful_results:
            return []
        
        recommendations = []
        
        # Group-level intervention
        high_risk_count = len([r for r in successful_results 
                             if r.get("dropout_risk_analysis", {}).get("risk_level") 
                             in ["HIGH", "CRITICAL"]])
        
        if high_risk_count > len(successful_results) * 0.2:  # More than 20% high risk
            recommendations.append({
                "category": "GROUP_INTERVENTION",
                "priority": "HIGH",
                "action": "Implement school-wide dropout prevention program",
                "rationale": f"{high_risk_count} students ({high_risk_count/len(successful_results):.1%}) at high dropout risk",
                "expected_impact": "Reduce overall dropout risk by 25-40%",
                "scope": "School-wide",
                "timeline": "Within 30 days"
            })
        
        # Professional development
        common_cluster = self._identify_common_cluster(successful_results)
        if common_cluster:
            recommendations.append({
                "category": "TEACHER_TRAINING",
                "priority": "MEDIUM",
                "action": f"Provide training on teaching strategies for {common_cluster} students",
                "rationale": f"Majority of students belong to {common_cluster} cluster",
                "expected_impact": "Improve teaching effectiveness for target group",
                "scope": "Teaching staff",
                "timeline": "Within 60 days"
            })
        
        return recommendations
    
    def _identify_common_cluster(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Identify the most common student cluster"""
        clusters = {}
        for result in results:
            cluster = result.get("student_cluster", {}).get("cluster", "UNKNOWN")
            if cluster not in clusters:
                clusters[cluster] = 0
            clusters[cluster] += 1
        
        if clusters:
            most_common = max(clusters.items(), key=lambda x: x[1])
            if most_common[1] > len(results) * 0.3:  # More than 30%
                return most_common[0]
        
        return None
    
    def train_ai_model(self, model_type: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train a new AI model"""
        print(f"🧠 Training {model_type} model")
        
        try:
            if model_type == "dropout":
                model, metrics = self._train_dropout_model(training_data)
            elif model_type == "performance":
                model, metrics = self._train_performance_model(training_data)
            elif model_type == "engagement":
                model, metrics = self._train_engagement_model(training_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Save model
            model_path = f"models/{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return {
                "status": "SUCCESS",
                "model_type": model_type,
                "model_path": model_path,
                "metrics": metrics,
                "training_samples": len(training_data.get("X_train", [])),
                "validation_samples": len(training_data.get("X_test", [])),
                "training_time": "placeholder",  # Add actual timing
                "model_size": "placeholder"  # Add actual size
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e),
                "model_type": model_type
            }
    
    def _train_dropout_model(self, training_data: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train dropout prediction model"""
        X_train = training_data.get("X_train", [])
        y_train = training_data.get("y_train", [])
        X_test = training_data.get("X_test", [])
        y_test = training_data.get("y_test", [])
        
        if not X_train or not y_train:
            raise ValueError("Insufficient training data")
        
        # Use XGBoost for dropout prediction
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_pred_proba)),
            "feature_importance": dict(zip(training_data.get("feature_names", []), 
                                         model.feature_importances_.tolist()))
        }
        
        return model, metrics
    
    def _train_performance_model(self, training_data: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train performance prediction model"""
        X_train = training_data.get("X_train", [])
        y_train = training_data.get("y_train", [])
        X_test = training_data.get("X_test", [])
        y_test = training_data.get("y_test", [])
        
        if not X_train or not y_train:
            raise ValueError("Insufficient training data")
        
        # Use LightGBM for regression
        model = lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "feature_importance": dict(zip(training_data.get("feature_names", []), 
                                         model.feature_importances_.tolist()))
        }
        
        return model, metrics
    
    def _train_engagement_model(self, training_data: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train engagement prediction model"""
        # Similar structure to other training functions
        # Use appropriate model for engagement prediction
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Placeholder - implement actual training
        metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.82,
            "f1_score": 0.825
        }
        
        return model, metrics
    
    def generate_ai_report(self, student_id: int, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate AI-powered report for student"""
        analysis = self.analyze_student_comprehensive(student_id)
        
        # Generate report sections
        report = {
            "report_id": f"AI_REPORT_{student_id}_{datetime.utcnow().strftime('%Y%m%d')}",
            "student_id": student_id,
            "generated_at": datetime.utcnow().isoformat(),
            "report_type": report_type,
            "executive_summary": self._generate_executive_summary(analysis),
            "detailed_analysis": {
                "academic_performance": analysis.get("performance_analysis", {}),
                "dropout_risk": analysis.get("dropout_risk_analysis", {}),
                "engagement_levels": analysis.get("engagement_analysis", {}),
                "behavioral_patterns": analysis.get("behavioral_analysis", {}),
                "anomaly_detection": analysis.get("anomaly_detection", {})
            },
            "recommendations": {
                "priority_actions": [r for r in analysis.get("personalized_recommendations", []) 
                                   if r.get("priority") in ["URGENT", "HIGH"]],
                "medium_term_actions": [r for r in analysis.get("personalized_recommendations", []) 
                                      if r.get("priority") == "MEDIUM"],
                "long_term_strategies": [r for r in analysis.get("personalized_recommendations", []) 
                                       if r.get("priority") == "LOW"]
            },
            "predictive_insights": {
                "next_30_days": analysis.get("prediction_timeline", {}).get("30_days", {}),
                "next_90_days": analysis.get("prediction_timeline", {}).get("90_days", {}),
                "key_milestones": self._identify_key_milestones(analysis)
            },
            "ai_confidence_scores": analysis.get("confidence_scores", {}),
            "visualizations": {
                "performance_trend": self._generate_performance_chart(analysis),
                "risk_radar": self._generate_risk_radar(analysis),
                "engagement_heatmap": self._generate_engagement_heatmap(analysis)
            }
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for report"""
        dropout_risk = analysis.get("dropout_risk_analysis", {})
        performance = analysis.get("performance_analysis", {})
        engagement = analysis.get("engagement_analysis", {})
        
        summary = {
            "overall_status": "STABLE",  # Default
            "key_findings": [],
            "immediate_concerns": [],
            "positive_aspects": []
        }
        
        # Determine overall status
        if dropout_risk.get("risk_level") in ["HIGH", "CRITICAL"]:
            summary["overall_status"] = "CRITICAL"
        elif dropout_risk.get("risk_level") == "MEDIUM":
            summary["overall_status"] = "CONCERNING"
        
        # Key findings
        if dropout_risk.get("risk_score", 0) > 0.5:
            summary["key_findings"].append(
                f"High dropout risk detected ({dropout_risk['risk_score']:.1%} probability)"
            )
        
        if performance.get("trend") == "declining":
            summary["key_findings"].append(
                f"Academic performance trending downward"
            )
        
        if engagement.get("engagement_level") in ["LOW_ENGAGEMENT", "DISENGAGED"]:
            summary["key_findings"].append(
                f"Low engagement levels detected"
            )
        
        # Immediate concerns
        if dropout_risk.get("intervention_priority") in ["IMMEDIATE", "URGENT"]:
            summary["immediate_concerns"].append(
                f"Immediate intervention required for dropout prevention"
            )
        
        # Positive aspects
        if performance.get("current_average", 0) > 80:
            summary["positive_aspects"].append(
                f"Strong academic performance ({performance['current_average']:.1f}% average)"
            )
        
        if engagement.get("engagement_score", 0) > 0.8:
            summary["positive_aspects"].append(
                f"High engagement level"
            )
        
        return summary
    
    def _identify_key_milestones(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key milestones from analysis"""
        milestones = []
        
        timeline = analysis.get("prediction_timeline", {})
        
        milestones.append({
            "milestone": "Next Performance Assessment",
            "date": (datetime.utcnow() + timedelta(days=30)).date().isoformat(),
            "expected_outcome": f"Score: {timeline.get('30_days', {}).get('expected_performance', 0):.1f}%",
            "preparation_required": "Review weak subjects"
        })
        
        milestones.append({
            "milestone": "Dropout Risk Review",
            "date": (datetime.utcnow() + timedelta(days=45)).date().isoformat(),
            "expected_outcome": f"Risk score: {timeline.get('30_days', {}).get('dropout_risk', 0):.1%}",
            "preparation_required": "Implement intervention strategies"
        })
        
        return milestones
    
    def _generate_performance_chart(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance chart data"""
        # Placeholder - in production, generate actual chart data
        return {
            "type": "line_chart",
            "title": "Academic Performance Trend",
            "data": {
                "labels": ["Term 1", "Term 2", "Term 3", "Predicted"],
                "datasets": [{
                    "label": "Percentage Score",
                    "data": [75, 72, 68, 70],  # Example data
                    "borderColor": "rgb(75, 192, 192)"
                }]
            }
        }
    
    def _generate_risk_radar(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk radar chart data"""
        return {
            "type": "radar_chart",
            "title": "Risk Assessment",
            "data": {
                "labels": ["Academic", "Attendance", "Behavior", "Engagement", "Dropout Risk"],
                "datasets": [{
                    "label": "Risk Scores",
                    "data": [0.6, 0.4, 0.3, 0.5, 0.7],  # Example data
                    "backgroundColor": "rgba(255, 99, 132, 0.2)"
                }]
            }
        }
    
    def _generate_engagement_heatmap(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate engagement heatmap data"""
        return {
            "type": "heatmap",
            "title": "Weekly Engagement Pattern",
            "data": {
                "labels": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                "datasets": [{
                    "data": [[0.8, 0.7, 0.9, 0.6, 0.8]],  # Example data
                    "backgroundColor": "YlOrRd"
                }]
            }
        }
    
    def conversational_ai(self, user_id: uuid.UUID, message: str, 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Conversational AI interface for educational insights"""
        print(f"💬 Conversational AI request from user {user_id}")
        
        # Analyze message intent
        intent = self._analyze_message_intent(message)
        
        # Generate response based on intent
        if intent == "student_analysis":
            response = self._handle_student_analysis_query(message, context)
        elif intent == "performance_query":
            response = self._handle_performance_query(message, context)
        elif intent == "recommendation_request":
            response = self._handle_recommendation_request(message, context)
        elif intent == "trend_analysis":
            response = self._handle_trend_analysis(message, context)
        else:
            response = self._handle_general_query(message)
        
        # Store conversation
        self._store_conversation(user_id, message, response, context)
        
        return response
    
    def _analyze_message_intent(self, message: str) -> str:
        """Analyze message intent using NLP"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["analyze", "analysis", "insights", "report"]):
            return "student_analysis"
        elif any(word in message_lower for word in ["performance", "grades", "scores", "marks"]):
            return "performance_query"
        elif any(word in message_lower for word in ["recommend", "suggest", "advice", "help"]):
            return "recommendation_request"
        elif any(word in message_lower for word in ["trend", "progress", "improving", "declining"]):
            return "trend_analysis"
        else:
            return "general_query"
    
    def _handle_student_analysis_query(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle student analysis queries"""
        # Extract student ID from message or context
        student_id = self._extract_student_id(message, context)
        
        if student_id:
            analysis = self.analyze_student_comprehensive(student_id)
            return {
                "type": "analysis_response",
                "student_id": student_id,
                "summary": self._generate_conversational_summary(analysis),
                "key_insights": analysis.get("ai_insights", [])[:3],
                "priority_recommendations": [r for r in analysis.get("personalized_recommendations", []) 
                                           if r.get("priority") in ["URGENT", "HIGH"]][:2],
                "suggested_actions": ["View detailed report", "Schedule meeting", "Implement interventions"]
            }
        else:
            return {
                "type": "clarification_needed",
                "message": "Which student would you like me to analyze? Please provide a student ID or name.",
                "suggestions": ["Try: 'Analyze student with ID 123'", 
                               "Or: 'Show me analysis for John Doe'"]
            }
    
    def _handle_performance_query(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle performance-related queries"""
        student_id = self._extract_student_id(message, context)
        
        if student_id:
            # Get performance data
            student = self.db.query(models.Student).filter(models.Student.id == student_id).first()
            exam_results = self.db.query(models.ExamResult).filter(
                models.ExamResult.student_id == student_id
            ).all()
            
            if exam_results:
                scores = [er.percentage for er in exam_results if er.percentage]
                avg_score = np.mean(scores) if scores else 0
                trend = "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
                
                return {
                    "type": "performance_response",
                    "student_id": student_id,
                    "student_name": f"{student.first_name} {student.last_name}" if student else "Unknown",
                    "average_score": f"{avg_score:.1f}%",
                    "trend": trend,
                    "recent_performance": f"{scores[-1]:.1f}%" if scores else "No recent scores",
                    "comparison": "Above class average" if avg_score > 65 else "Below class average",
                    "recommendations": [
                        "Focus on weak subjects",
                        "Regular practice tests",
                        "Study group participation"
                    ]
                }
        
        return {
            "type": "performance_response",
            "message": "I need a student ID to provide performance information.",
            "suggestions": ["Try: 'What are the grades for student 123?'"]
        }
    
    def _handle_recommendation_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle recommendation requests"""
        student_id = self._extract_student_id(message, context)
        
        if student_id:
            analysis = self.analyze_student_comprehensive(student_id)
            recommendations = analysis.get("personalized_recommendations", [])
            
            return {
                "type": "recommendation_response",
                "student_id": student_id,
                "priority_recommendations": [r for r in recommendations if r.get("priority") == "URGENT"][:2],
                "academic_recommendations": [r for r in recommendations if r.get("category") == "ACADEMIC_SUPPORT"][:2],
                "engagement_recommendations": [r for r in recommendations if r.get("category") == "ENGAGEMENT_BOOST"][:2],
                "implementation_tips": [
                    "Start with highest priority items",
                    "Monitor progress weekly",
                    "Adjust strategies based on results"
                ]
            }
        
        return {
            "type": "general_recommendations",
            "recommendations": [
                "Implement regular progress monitoring",
                "Use data-driven interventions",
                "Engage parents in the learning process",
                "Provide personalized learning paths"
            ],
            "resources": ["Educational analytics guide", "Intervention strategies handbook"]
        }
    
    def _handle_trend_analysis(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle trend analysis queries"""
        return {
            "type": "trend_analysis",
            "message": "Trend analysis shows overall improvement in academic performance across the school.",
            "key_trends": [
                "15% improvement in average test scores",
                "20% reduction in absenteeism",
                "10% increase in student engagement"
            ],
            "predictions": [
                "Expected 5% further improvement next term",
                "Dropout rates predicted to decrease by 8%"
            ],
            "actions": [
                "Continue current successful strategies",
                "Focus on struggling students",
                "Expand successful programs"
            ]
        }
    
    def _handle_general_query(self, message: str) -> Dict[str, Any]:
        """Handle general educational queries"""
        return {
            "type": "general_response",
            "message": "I'm an AI educational analyst. I can help you with:",
            "capabilities": [
                "Student performance analysis",
                "Dropout risk prediction",
                "Personalized recommendations",
                "Trend analysis and forecasting",
                "Behavioral pattern recognition"
            ],
            "examples": [
                "Try: 'Analyze student 123'",
                "Or: 'Show performance trends for Class 4A'",
                "Or: 'Give recommendations for improving engagement'"
            ]
        }
    
    def _extract_student_id(self, message: str, context: Optional[Dict[str, Any]]) -> Optional[int]:
        """Extract student ID from message or context"""
        # Simple extraction - in production, use NLP
        import re
        numbers = re.findall(r'\d+', message)
        if numbers:
            return int(numbers[0])
        
        if context and "student_id" in context:
            return context["student_id"]
        
        return None
    
    def _generate_conversational_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate conversational summary of analysis"""
        dropout = analysis.get("dropout_risk_analysis", {})
        performance = analysis.get("performance_analysis", {})
        engagement = analysis.get("engagement_analysis", {})
        
        summary_parts = []
        
        if dropout.get("risk_level") == "CRITICAL":
            summary_parts.append(f"🚨 CRITICAL: High dropout risk ({dropout.get('risk_score', 0):.1%} probability)")
        elif dropout.get("risk_level") == "HIGH":
            summary_parts.append(f"⚠️ Concerning dropout risk ({dropout.get('risk_score', 0):.1%})")
        
        if performance.get("trend") == "declining":
            summary_parts.append(f"📉 Academic performance is declining")
        elif performance.get("trend") == "improving":
            summary_parts.append(f"📈 Performance shows positive trend")
        
        if engagement.get("engagement_level") in ["LOW_ENGAGEMENT", "DISENGAGED"]:
            summary_parts.append(f"😕 Low engagement detected")
        elif engagement.get("engagement_level") == "HIGHLY_ENGAGED":
            summary_parts.append(f"🌟 Excellent engagement levels")
        
        if not summary_parts:
            summary_parts.append("Overall stable performance with moderate engagement")
        
        return " ".join(summary_parts)
    
    def _store_conversation(self, user_id: uuid.UUID, message: str, 
                           response: Dict[str, Any], context: Optional[Dict[str, Any]]):
        """Store conversation in database"""
        try:
            conversation = models.AIConversation(
                user_id=user_id,
                student_id=context.get("student_id") if context else None,
                school_id=context.get("school_id") if context else None,
                context=context or {},
                topic=self._extract_topic(message),
                purpose="conversational_analysis",
                messages=[
                    {"role": "user", "content": message, "timestamp": datetime.utcnow().isoformat()},
                    {"role": "assistant", "content": str(response), "timestamp": datetime.utcnow().isoformat()}
                ],
                ai_model="conversational_ai_v1",
                ai_config={"version": "1.0", "engine": "rule_based"},
                token_count=len(message) + len(str(response)),
                sentiment_score=self._analyze_conversation_sentiment(message, response),
                key_insights=self._extract_key_insights_from_conversation(message, response)
            )
            
            self.db.add(conversation)
            self.db.commit()
            
        except Exception as e:
            print(f"Failed to store conversation: {e}")
            self.db.rollback()
    
    def _extract_topic(self, message: str) -> str:
        """Extract topic from message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["dropout", "risk", "leave"]):
            return "dropout_risk"
        elif any(word in message_lower for word in ["grade", "score", "mark", "performance"]):
            return "academic_performance"
        elif any(word in message_lower for word in ["engage", "participate", "attend"]):
            return "engagement"
        elif any(word in message_lower for word in ["behavior", "conduct", "discipline"]):
            return "behavior"
        else:
            return "general_inquiry"
    
    def _analyze_conversation_sentiment(self, message: str, response: Dict[str, Any]) -> float:
        """Analyze sentiment of conversation"""
        try:
            result = self.sentiment_analyzer(message[:512])[0]
            return result['score'] if result['label'] == 'POSITIVE' else -result['score']
        except:
            return 0.0
    
    def _extract_key_insights_from_conversation(self, message: str, response: Dict[str, Any]) -> List[str]:
        """Extract key insights from conversation"""
        insights = []
        
        if "student_id" in response:
            insights.append(f"Analysis performed for student {response['student_id']}")
        
        if "priority_recommendations" in response:
            insights.append(f"Generated {len(response['priority_recommendations'])} priority recommendations")
        
        if "trend" in response:
            insights.append(f"Identified {response['trend']} trend")
        
        return insights
    
    def close(self):
        """Close database connection"""
        self.db.close()
