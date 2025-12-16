from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import uuid

import models
from schemas import (
    SchoolCreate, SchoolUpdate, StudentCreate, StudentUpdate,
    ExamCreate, ExamResultCreate, LearningPathCreate,
    AIPredictionCreate, AIModelCreate, AITrainingJobCreate
)

class CRUDService:
    def __init__(self, db: Session):
        self.db = db
    
    # School CRUD
    def create_school(self, school_data: SchoolCreate) -> models.School:
        """Create new school"""
        school = models.School(**school_data.model_dump())
        self.db.add(school)
        self.db.commit()
        self.db.refresh(school)
        return school
    
    def get_school(self, school_id: int) -> Optional[models.School]:
        """Get school by ID"""
        return self.db.query(models.School).filter(models.School.id == school_id).first()
    
    def get_schools(
        self, 
        skip: int = 0, 
        limit: int = 100,
        county: Optional[str] = None,
        level: Optional[str] = None
    ) -> List[models.School]:
        """Get schools with filters"""
        query = self.db.query(models.School)
        
        if county:
            query = query.filter(models.School.county == county)
        if level:
            query = query.filter(models.School.level == level)
        
        return query.offset(skip).limit(limit).all()
    
    def update_school(self, school_id: int, school_data: SchoolUpdate) -> Optional[models.School]:
        """Update school"""
        school = self.get_school(school_id)
        if not school:
            return None
        
        update_data = school_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(school, field, value)
        
        school.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(school)
        return school
    
    def delete_school(self, school_id: int) -> bool:
        """Delete school"""
        school = self.get_school(school_id)
        if not school:
            return False
        
        self.db.delete(school)
        self.db.commit()
        return True
    
    # Student CRUD
    def create_student(self, student_data: StudentCreate) -> models.Student:
        """Create new student"""
        student = models.Student(**student_data.model_dump())
        self.db.add(student)
        self.db.commit()
        self.db.refresh(student)
        
        # Update school student count
        school = self.get_school(student.school_id)
        if school:
            school.total_students += 1
            self.db.commit()
        
        return student
    
    def get_student(self, student_id: int) -> Optional[models.Student]:
        """Get student by ID"""
        return self.db.query(models.Student).filter(models.Student.id == student_id).first()
    
    def get_students(
        self,
        school_id: Optional[int] = None,
        class_id: Optional[int] = None,
        is_active: Optional[bool] = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[models.Student]:
        """Get students with filters"""
        query = self.db.query(models.Student)
        
        if school_id:
            query = query.filter(models.Student.school_id == school_id)
        if class_id:
            query = query.filter(models.Student.class_id == class_id)
        if is_active is not None:
            query = query.filter(models.Student.is_active == is_active)
        
        return query.offset(skip).limit(limit).all()
    
    def update_student(self, student_id: int, student_data: StudentUpdate) -> Optional[models.Student]:
        """Update student"""
        student = self.get_student(student_id)
        if not student:
            return None
        
        update_data = student_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(student, field, value)
        
        student.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(student)
        return student
    
    def delete_student(self, student_id: int) -> bool:
        """Delete student (soft delete)"""
        student = self.get_student(student_id)
        if not student:
            return False
        
        student.is_active = False
        student.status = "inactive"
        student.updated_at = datetime.utcnow()
        
        # Update school student count
        school = self.get_school(student.school_id)
        if school and school.total_students > 0:
            school.total_students -= 1
        
        self.db.commit()
        return True
    
    # Exam CRUD
    def create_exam(self, exam_data: ExamCreate) -> models.Exam:
        """Create new exam"""
        exam = models.Exam(**exam_data.model_dump())
        self.db.add(exam)
        self.db.commit()
        self.db.refresh(exam)
        return exam
    
    def get_exam(self, exam_id: int) -> Optional[models.Exam]:
        """Get exam by ID"""
        return self.db.query(models.Exam).filter(models.Exam.id == exam_id).first()
    
    def get_exams(
        self,
        school_id: Optional[int] = None,
        class_id: Optional[int] = None,
        year: Optional[int] = None,
        term: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[models.Exam]:
        """Get exams with filters"""
        query = self.db.query(models.Exam)
        
        if school_id:
            query = query.filter(models.Exam.school_id == school_id)
        if class_id:
            query = query.filter(models.Exam.class_id == class_id)
        if year:
            query = query.filter(models.Exam.year == year)
        if term:
            query = query.filter(models.Exam.term == term)
        
        return query.order_by(desc(models.Exam.conducted_date)).offset(skip).limit(limit).all()
    
    # Exam Result CRUD
    def create_exam_result(self, result_data: ExamResultCreate) -> models.ExamResult:
        """Create exam result"""
        # Calculate percentage if not provided
        if result_data.percentage is None:
            result_data.percentage = (result_data.raw_score / result_data.total_score) * 100
        
        result = models.ExamResult(**result_data.model_dump())
        self.db.add(result)
        self.db.commit()
        
        # Update exam statistics
        self._update_exam_statistics(result.exam_id)
        
        self.db.refresh(result)
        return result
    
    def get_exam_results(
        self,
        student_id: Optional[int] = None,
        exam_id: Optional[int] = None,
        subject_id: Optional[int] = None,
        year: Optional[int] = None,
        term: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[models.ExamResult]:
        """Get exam results with filters"""
        query = self.db.query(models.ExamResult)
        
        if student_id:
            query = query.filter(models.ExamResult.student_id == student_id)
        if exam_id:
            query = query.filter(models.ExamResult.exam_id == exam_id)
        if subject_id:
            query = query.filter(models.ExamResult.subject_id == subject_id)
        if year:
            query = query.filter(models.ExamResult.year == year)
        if term:
            query = query.filter(models.ExamResult.term == term)
        
        return query.order_by(desc(models.ExamResult.created_at)).offset(skip).limit(limit).all()
    
    def _update_exam_statistics(self, exam_id: int):
        """Update exam statistics after new results"""
        results = self.db.query(models.ExamResult).filter(
            models.ExamResult.exam_id == exam_id
        ).all()
        
        if not results:
            return
        
        exam = self.get_exam(exam_id)
        if not exam:
            return
        
        # Calculate basic statistics
        scores = [r.percentage for r in results if r.percentage is not None]
        
        if scores:
            import numpy as np
            exam.difficulty_index = float(np.mean(scores) / 100)  # Normalize to 0-1
            
            # Simple discrimination index (top 27% vs bottom 27%)
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            top_count = int(n * 0.27)
            bottom_count = int(n * 0.27)
            
            if top_count > 0 and bottom_count > 0:
                top_avg = np.mean(sorted_scores[-top_count:])
                bottom_avg = np.mean(sorted_scores[:bottom_count])
                exam.discrimination_index = float((top_avg - bottom_avg) / 100)
            
            exam.analysis_complete = True
        
        self.db.commit()
    
    # Attendance CRUD
    def record_attendance(
        self,
        student_id: int,
        date: date,
        status: str,
        reason: Optional[str] = None,
        session: str = "full_day"
    ) -> models.Attendance:
        """Record attendance"""
        # Check if attendance already recorded for this date
        existing = self.db.query(models.Attendance).filter(
            models.Attendance.student_id == student_id,
            models.Attendance.date == date,
            models.Attendance.session == session
        ).first()
        
        if existing:
            existing.status = status
            existing.reason = reason
            self.db.commit()
            self.db.refresh(existing)
            return existing
        
        attendance = models.Attendance(
            student_id=student_id,
            date=date,
            status=status,
            reason=reason,
            session=session
        )
        
        self.db.add(attendance)
        self.db.commit()
        self.db.refresh(attendance)
        return attendance
    
    def get_attendance(
        self,
        student_id: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[models.Attendance]:
        """Get attendance records"""
        query = self.db.query(models.Attendance)
        
        if student_id:
            query = query.filter(models.Attendance.student_id == student_id)
        if start_date:
            query = query.filter(models.Attendance.date >= start_date)
        if end_date:
            query = query.filter(models.Attendance.date <= end_date)
        if status:
            query = query.filter(models.Attendance.status == status)
        
        return query.order_by(desc(models.Attendance.date)).offset(skip).limit(limit).all()
    
    # AI Prediction CRUD
    def create_ai_prediction(self, prediction_data: AIPredictionCreate) -> models.AIPrediction:
        """Create AI prediction"""
        prediction = models.AIPrediction(**prediction_data.model_dump())
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction
    
    def get_ai_predictions(
        self,
        student_id: Optional[int] = None,
        model_id: Optional[int] = None,
        prediction_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[models.AIPrediction]:
        """Get AI predictions"""
        query = self.db.query(models.AIPrediction)
        
        if student_id:
            query = query.filter(models.AIPrediction.student_id == student_id)
        if model_id:
            query = query.filter(models.AIPrediction.model_id == model_id)
        if prediction_type:
            query = query.filter(models.AIPrediction.prediction_type == prediction_type)
        if start_date:
            query = query.filter(models.AIPrediction.prediction_date >= start_date)
        if end_date:
            query = query.filter(models.AIPrediction.prediction_date <= end_date)
        
        return query.order_by(desc(models.AIPrediction.created_at)).offset(skip).limit(limit).all()
    
    # Learning Path CRUD
    def create_learning_path(self, path_data: LearningPathCreate) -> models.LearningPath:
        """Create learning path"""
        path = models.LearningPath(**path_data.model_dump())
        self.db.add(path)
        self.db.commit()
        self.db.refresh(path)
        return path
    
    def get_learning_paths(
        self,
        student_id: Optional[int] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[models.LearningPath]:
        """Get learning paths"""
        query = self.db.query(models.LearningPath)
        
        if student_id:
            query = query.filter(models.LearningPath.student_id == student_id)
        if status:
            query = query.filter(models.LearningPath.status == status)
        
        return query.order_by(desc(models.LearningPath.created_at)).offset(skip).limit(limit).all()
    
    def update_learning_path_progress(self, path_id: int, progress: float) -> Optional[models.LearningPath]:
        """Update learning path progress"""
        path = self.db.query(models.LearningPath).filter(models.LearningPath.id == path_id).first()
        if not path:
            return None
        
        path.progress = progress
        if progress >= 100:
            path.status = "completed"
            path.completion_rate = 100.0
        
        path.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(path)
        return path
    
    # Analytics Queries
    def get_class_performance(
        self,
        class_id: int,
        term: Optional[str] = None,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get class performance analytics"""
        # Get all students in class
        students = self.db.query(models.Student).filter(
            models.Student.class_id == class_id,
            models.Student.is_active == True
        ).all()
        
        student_ids = [s.id for s in students]
        
        # Get exam results
        query = self.db.query(models.ExamResult).filter(
            models.ExamResult.student_id.in_(student_ids)
        )
        
        if term:
            query = query.filter(models.ExamResult.term == term)
        if year:
            query = query.filter(models.ExamResult.year == year)
        
        results = query.all()
        
        if not results:
            return {"status": "no_data", "message": "No exam results found"}
        
        # Calculate statistics
        import numpy as np
        scores = [r.percentage for r in results if r.percentage is not None]
        
        performance_by_subject = {}
        for result in results:
            if result.subject and result.percentage is not None:
                subject_name = result.subject.name
                if subject_name not in performance_by_subject:
                    performance_by_subject[subject_name] = []
                performance_by_subject[subject_name].append(result.percentage)
        
        subject_averages = {
            subject: round(np.mean(scores), 2)
            for subject, scores in performance_by_subject.items()
        }
        
        # Identify top performers and those needing attention
        student_scores = {}
        for student in students:
            student_results = [r for r in results if r.student_id == student.id]
            if student_results:
                student_scores[student.id] = {
                    "student": student,
                    "average": np.mean([r.percentage for r in student_results if r.percentage is not None]),
                    "results": student_results
                }
        
        top_performers = sorted(
            [s for s in student_scores.values()],
            key=lambda x: x["average"],
            reverse=True
        )[:5]
        
        needs_attention = sorted(
            [s for s in student_scores.values()],
            key=lambda x: x["average"]
        )[:5]
        
        return {
            "class_id": class_id,
            "total_students": len(students),
            "average_score": round(np.mean(scores), 2) if scores else 0,
            "highest_score": round(max(scores), 2) if scores else 0,
            "lowest_score": round(min(scores), 2) if scores else 0,
            "standard_deviation": round(np.std(scores), 2) if scores else 0,
            "subject_performance": subject_averages,
            "top_performers": [
                {
                    "student_id": s["student"].id,
                    "name": f"{s['student'].first_name} {s['student'].last_name}",
                    "average_score": round(s["average"], 2)
                }
                for s in top_performers
            ],
            "needs_attention": [
                {
                    "student_id": s["student"].id,
                    "name": f"{s['student'].first_name} {s['student'].last_name}",
                    "average_score": round(s["average"], 2),
                    "weak_subjects": self._identify_weak_subjects(s["results"])
                }
                for s in needs_attention
            ]
        }
    
    def _identify_weak_subjects(self, results: List[models.ExamResult]) -> List[Dict[str, Any]]:
        """Identify weak subjects from results"""
        if not results:
            return []
        
        subject_scores = {}
        for result in results:
            if result.subject and result.percentage is not None:
                subject_name = result.subject.name
                if subject_name not in subject_scores:
                    subject_scores[subject_name] = []
                subject_scores[subject_name].append(result.percentage)
        
        weak_subjects = []
        for subject, scores in subject_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 50:  # Threshold for weak subject
                weak_subjects.append({
                    "subject": subject,
                    "average_score": round(avg_score, 2),
                    "recommendation": "Needs additional support"
                })
        
        return weak_subjects
    
    def get_student_progress_timeline(
        self,
        student_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get student progress timeline"""
        # Get exam results
        query = self.db.query(models.ExamResult).filter(
            models.ExamResult.student_id == student_id
        )
        
        if start_date:
            # Filter by exam date through join
            query = query.join(models.Exam).filter(models.Exam.conducted_date >= start_date)
        if end_date:
            query = query.join(models.Exam).filter(models.Exam.conducted_date <= end_date)
        
        results = query.order_by(asc(models.Exam.conducted_date)).all()
        
        # Get attendance
        attendance = self.get_attendance(
            student_id=student_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get behaviors
        behaviors = self.db.query(models.StudentBehavior).filter(
            models.StudentBehavior.student_id == student_id
        )
        
        if start_date:
            behaviors = behaviors.filter(models.StudentBehavior.date_observed >= start_date)
        if end_date:
            behaviors = behaviors.filter(models.StudentBehavior.date_observed <= end_date)
        
        behaviors = behaviors.order_by(asc(models.StudentBehavior.date_observed)).all()
        
        # Format timeline
        timeline = []
        
        # Exam events
        for result in results:
            if result.exam:
                timeline.append({
                    "date": result.exam.conducted_date or result.created_at.date(),
                    "type": "exam",
                    "title": result.exam.name,
                    "description": f"{result.subject.name if result.subject else 'Unknown'}: {result.percentage}%",
                    "data": {
                        "score": result.percentage,
                        "grade": result.grade,
                        "position": result.position
                    }
                })
        
        # Attendance events (only significant absences)
        significant_absences = [a for a in attendance if a.status == "absent" and not a.reason]
        for absence in significant_absences[:10]:  # Limit to 10 most recent
            timeline.append({
                "date": absence.date,
                "type": "attendance",
                "title": "Absence",
                "description": "Unexcused absence",
                "data": {"status": "absent", "session": absence.session}
            })
        
        # Behavior events
        for behavior in behaviors[:10]:  # Limit to 10 most recent
            timeline.append({
                "date": behavior.date_observed or behavior.created_at.date(),
                "type": "behavior",
                "title": f"{behavior.behavior_type.capitalize()} Behavior",
                "description": behavior.description[:100],
                "data": {
                    "type": behavior.behavior_type,
                    "severity": behavior.severity,
                    "category": behavior.category
                }
            })
        
        # Sort timeline by date
        timeline.sort(key=lambda x: x["date"])
        
        return {
            "student_id": student_id,
            "start_date": start_date,
            "end_date": end_date,
            "total_events": len(timeline),
            "timeline": timeline,
            "summary": self._generate_timeline_summary(timeline)
        }
    
    def _generate_timeline_summary(self, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary from timeline"""
        exam_scores = [event["data"]["score"] for event in timeline if event["type"] == "exam" and "score" in event["data"]]
        absences = len([event for event in timeline if event["type"] == "attendance"])
        positive_behaviors = len([event for event in timeline if event["type"] == "behavior" and event["data"]["type"] == "positive"])
        negative_behaviors = len([event for event in timeline if event["type"] == "behavior" and event["data"]["type"] == "negative"])
        
        import numpy as np
        summary = {
            "exam_count": len(exam_scores),
            "average_exam_score": round(np.mean(exam_scores), 2) if exam_scores else 0,
            "absence_count": absences,
            "behavior_ratio": positive_behaviors / (negative_behaviors + 1) if negative_behaviors > 0 else positive_behaviors,
            "trend": self._calculate_timeline_trend(timeline)
        }
        
        return summary
    
    def _calculate_timeline_trend(self, timeline: List[Dict[str, Any]]) -> str:
        """Calculate overall trend from timeline"""
        if len(timeline) < 2:
            return "insufficient_data"
        
        # Get exam scores in chronological order
        exam_events = [event for event in timeline if event["type"] == "exam"]
        if len(exam_events) < 2:
            return "stable"
        
        scores = [event["data"]["score"] for event in exam_events if "score" in event["data"]]
        if len(scores) < 2:
            return "stable"
        
        # Simple trend calculation
        if scores[-1] > scores[0] + 5:
            return "improving"
        elif scores[-1] < scores[0] - 5:
            return "declining"
        else:
            return "stable"
    
    # Dashboard Statistics
    def get_dashboard_stats(self, school_id: Optional[int] = None) -> Dict[str, Any]:
        """Get dashboard statistics"""
        stats = {}
        
        # Student statistics
        student_query = self.db.query(models.Student)
        if school_id:
            student_query = student_query.filter(models.Student.school_id == school_id)
        
        total_students = student_query.count()
        active_students = student_query.filter(models.Student.is_active == True).count()
        
        stats["students"] = {
            "total": total_students,
            "active": active_students,
            "inactive": total_students - active_students
        }
        
        # Teacher statistics
        teacher_query = self.db.query(models.Teacher)
        if school_id:
            teacher_query = teacher_query.filter(models.Teacher.school_id == school_id)
        
        stats["teachers"] = {
            "total": teacher_query.count(),
            "active": teacher_query.filter(models.Teacher.is_active == True).count()
        }
        
        # Attendance statistics (last 30 days)
        thirty_days_ago = datetime.utcnow().date() - timedelta(days=30)
        attendance_query = self.db.query(models.Attendance).filter(
            models.Attendance.date >= thirty_days_ago
        )
        
        if school_id:
            # Join with students to filter by school
            attendance_query = attendance_query.join(models.Student).filter(
                models.Student.school_id == school_id
            )
        
        attendance_records = attendance_query.all()
        
        if attendance_records:
            present_count = len([a for a in attendance_records if a.status == "present"])
            stats["attendance"] = {
                "rate": round(present_count / len(attendance_records) * 100, 2),
                "total_records": len(attendance_records)
            }
        else:
            stats["attendance"] = {"rate": 0, "total_records": 0}
        
        # Exam statistics (last term)
        exam_query = self.db.query(models.Exam)
        if school_id:
            exam_query = exam_query.filter(models.Exam.school_id == school_id)
        
        recent_exams = exam_query.order_by(desc(models.Exam.conducted_date)).limit(5).all()
        
        if recent_exams:
            exam_ids = [exam.id for exam in recent_exams]
            results = self.db.query(models.ExamResult).filter(
                models.ExamResult.exam_id.in_(exam_ids)
            ).all()
            
            if results:
                scores = [r.percentage for r in results if r.percentage is not None]
                stats["exams"] = {
                    "recent_count": len(recent_exams),
                    "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
                    "results_entered": len(results)
                }
        
        # AI Insights count
        ai_query = self.db.query(models.AIPrediction)
        if school_id:
            # Join with students to filter by school
            ai_query = ai_query.join(models.Student).filter(
                models.Student.school_id == school_id
            )
        
        stats["ai_insights"] = {
            "total_predictions": ai_query.count(),
            "last_7_days": ai_query.filter(
                models.AIPrediction.created_at >= datetime.utcnow() - timedelta(days=7)
            ).count()
        }
        
        # Interventions
        intervention_query = self.db.query(models.Intervention)
        if school_id:
            intervention_query = intervention_query.join(models.Student).filter(
                models.Student.school_id == school_id
            )
        
        stats["interventions"] = {
            "active": intervention_query.filter(models.Intervention.status == "active").count(),
            "completed": intervention_query.filter(models.Intervention.status == "completed").count(),
            "pending": intervention_query.filter(models.Intervention.status == "pending").count()
        }
        
        return stats
    
    # Search functionality
    def search_students(
        self,
        query: str,
        school_id: Optional[int] = None,
        limit: int = 20
    ) -> List[models.Student]:
        """Search students by name or admission number"""
        search_query = self.db.query(models.Student)
        
        if school_id:
            search_query = search_query.filter(models.Student.school_id == school_id)
        
        # Split query into terms
        terms = query.split()
        
        # Build search conditions
        conditions = []
        for term in terms:
            term_pattern = f"%{term}%"
            conditions.extend([
                models.Student.first_name.ilike(term_pattern),
                models.Student.last_name.ilike(term_pattern),
                models.Student.admission_number.ilike(term_pattern)
            ])
        
        search_query = search_query.filter(or_(*conditions))
        return search_query.limit(limit).all()
    
    # Bulk operations
    def bulk_create_exam_results(
        self,
        results_data: List[Dict[str, Any]],
        exam_id: int,
        subject_id: int
    ) -> List[models.ExamResult]:
        """Bulk create exam results"""
        created_results = []
        
        for result_data in results_data:
            try:
                result = models.ExamResult(
                    student_id=result_data["student_id"],
                    exam_id=exam_id,
                    subject_id=subject_id,
                    raw_score=result_data["raw_score"],
                    total_score=result_data.get("total_score", 100.0),
                    percentage=(result_data["raw_score"] / result_data.get("total_score", 100.0)) * 100,
                    grade=self._calculate_grade(result_data["raw_score"], result_data.get("total_score", 100.0)),
                    points=self._calculate_points(result_data["raw_score"], result_data.get("total_score", 100.0))
                )
                
                self.db.add(result)
                created_results.append(result)
                
            except Exception as e:
                print(f"Error creating result for student {result_data.get('student_id')}: {e}")
        
        self.db.commit()
        
        # Update exam statistics
        self._update_exam_statistics(exam_id)
        
        return created_results
    
    def _calculate_grade(self, raw_score: float, total_score: float = 100.0) -> str:
        """Calculate grade from score"""
        percentage = (raw_score / total_score) * 100
        
        if percentage >= 80:
            return "A"
        elif percentage >= 75:
            return "A-"
        elif percentage >= 70:
            return "B+"
        elif percentage >= 65:
            return "B"
        elif percentage >= 60:
            return "B-"
        elif percentage >= 55:
            return "C+"
        elif percentage >= 50:
            return "C"
        elif percentage >= 45:
            return "C-"
        elif percentage >= 40:
            return "D+"
        elif percentage >= 35:
            return "D"
        elif percentage >= 30:
            return "D-"
        else:
            return "E"
    
    def _calculate_points(self, raw_score: float, total_score: float = 100.0) -> int:
        """Calculate points from score"""
        percentage = (raw_score / total_score) * 100
        
        if percentage >= 80:
            return 12
        elif percentage >= 75:
            return 11
        elif percentage >= 70:
            return 10
        elif percentage >= 65:
            return 9
        elif percentage >= 60:
            return 8
        elif percentage >= 55:
            return 7
        elif percentage >= 50:
            return 6
        elif percentage >= 45:
            return 5
        elif percentage >= 40:
            return 4
        elif percentage >= 35:
            return 3
        elif percentage >= 30:
            return 2
        else:
            return 1
    
    # Data export
    def export_student_data(
        self,
        school_id: int,
        format: str = "csv"
    ) -> Dict[str, Any]:
        """Export student data"""
        students = self.db.query(models.Student).filter(
            models.Student.school_id == school_id,
            models.Student.is_active == True
        ).all()
        
        # Get exam results for each student
        student_data = []
        for student in students:
            results = self.db.query(models.ExamResult).filter(
                models.ExamResult.student_id == student.id
            ).all()
            
            student_info = {
                "admission_number": student.admission_number,
                "first_name": student.first_name,
                "last_name": student.last_name,
                "class": student.class_rel.name if student.class_rel else "",
                "attendance_rate": self._calculate_student_attendance_rate(student.id),
                "average_score": self._calculate_student_average_score(student.id),
                "dropout_risk_score": student.dropout_risk_score,
                "engagement_score": student.engagement_score,
                "status": student.status
            }
            
            # Add subject scores
            for result in results[:5]:  # Limit to 5 recent subjects
                if result.subject:
                    student_info[f"subject_{result.subject.code}"] = result.percentage
            
            student_data.append(student_info)
        
        return {
            "school_id": school_id,
            "exported_at": datetime.utcnow().isoformat(),
            "total_students": len(student_data),
            "format": format,
            "data": student_data
        }
    
    def _calculate_student_attendance_rate(self, student_id: int) -> float:
        """Calculate student attendance rate"""
        attendance = self.get_attendance(student_id=student_id)
        if not attendance:
            return 0.0
        
        present_count = len([a for a in attendance if a.status == "present"])
        return round(present_count / len(attendance) * 100, 2)
    
    def _calculate_student_average_score(self, student_id: int) -> float:
        """Calculate student average score"""
        results = self.get_exam_results(student_id=student_id)
        if not results:
            return 0.0
        
        scores = [r.percentage for r in results if r.percentage is not None]
        return round(sum(scores) / len(scores), 2) if scores else 0.0
