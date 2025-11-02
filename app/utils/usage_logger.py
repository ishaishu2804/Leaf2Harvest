"""
Usage Logger Utility for LeafToHarvest Application

This module handles logging of API usage data to the database, including
OpenAI API calls, local model usage, and fallback analysis. It provides
a clean interface for tracking user activity and costs.
"""

from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UsageLogger:
    """
    Utility class for logging API usage data to the database.
    
    This class provides methods for logging various types of API usage,
    including OpenAI calls, local model predictions, and fallback analyses.
    It automatically handles database connections and error handling.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize the usage logger with database connection.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("Usage logger initialized")
    
    def log_openai_usage(self, usage_data: Dict) -> bool:
        """
        Log OpenAI API usage to the database.
        
        Args:
            usage_data: Dictionary containing usage information:
                - user_id: ID of the user
                - timestamp: When the API call was made
                - model_name: OpenAI model used
                - prompt_tokens: Number of input tokens
                - completion_tokens: Number of output tokens
                - total_tokens: Total tokens used
                - estimated_cost: Calculated cost in USD
                - error: Error message if any
                
        Returns:
            True if logging was successful, False otherwise
        """
        try:
            session = self.Session()
            
            # Import here to avoid circular imports
            from app.models import ApiUsageLog
            
            # Create new usage log entry
            usage_log = ApiUsageLog(
                user_id=usage_data['user_id'],
                timestamp=usage_data['timestamp'],
                model_name=usage_data['model_name'],
                prompt_tokens=usage_data['prompt_tokens'],
                completion_tokens=usage_data['completion_tokens'],
                total_tokens=usage_data['total_tokens'],
                estimated_cost=usage_data['estimated_cost'],
                error_message=usage_data.get('error'),
                service_type='openai'
            )
            
            session.add(usage_log)
            session.commit()
            session.close()
            
            logger.info(f"OpenAI usage logged for user {usage_data['user_id']}: {usage_data['total_tokens']} tokens")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log OpenAI usage: {str(e)}")
            return False
    
    def log_local_model_usage(self, user_id: int, model_name: str, confidence_score: float = None) -> bool:
        """
        Log local model usage to the database.
        
        Args:
            user_id: ID of the user
            model_name: Name of the local model used
            confidence_score: Confidence score of the prediction
            
        Returns:
            True if logging was successful, False otherwise
        """
        try:
            session = self.Session()
            
            # Import here to avoid circular imports
            from app.models import ApiUsageLog
            
            # Create new usage log entry
            usage_log = ApiUsageLog(
                user_id=user_id,
                timestamp=datetime.utcnow(),
                model_name=model_name,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost=0.0,
                error_message=None,
                service_type='local_model',
                confidence_score=confidence_score
            )
            
            session.add(usage_log)
            session.commit()
            session.close()
            
            logger.info(f"Local model usage logged for user {user_id}: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log local model usage: {str(e)}")
            return False
    
    def log_fallback_usage(self, user_id: int, fallback_type: str = 'general') -> bool:
        """
        Log fallback analysis usage to the database.
        
        Args:
            user_id: ID of the user
            fallback_type: Type of fallback analysis performed
            
        Returns:
            True if logging was successful, False otherwise
        """
        try:
            session = self.Session()
            
            # Import here to avoid circular imports
            from app.models import ApiUsageLog
            
            # Create new usage log entry
            usage_log = ApiUsageLog(
                user_id=user_id,
                timestamp=datetime.utcnow(),
                model_name=f'fallback_{fallback_type}',
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost=0.0,
                error_message=None,
                service_type='fallback'
            )
            
            session.add(usage_log)
            session.commit()
            session.close()
            
            logger.info(f"Fallback usage logged for user {user_id}: {fallback_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log fallback usage: {str(e)}")
            return False
    
    def get_user_usage_stats(self, user_id: int, days: int = 30) -> Dict:
        """
        Get usage statistics for a specific user.
        
        Args:
            user_id: ID of the user
            days: Number of days to look back
            
        Returns:
            Dictionary containing usage statistics
        """
        try:
            session = self.Session()
            
            # Import here to avoid circular imports
            from app.models import ApiUsageLog
            from sqlalchemy import func
            from datetime import timedelta
            
            # Calculate date range
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Query usage logs for the user
            usage_logs = session.query(ApiUsageLog).filter(
                ApiUsageLog.user_id == user_id,
                ApiUsageLog.timestamp >= start_date
            ).all()
            
            # Calculate statistics
            total_calls = len(usage_logs)
            openai_calls = len([log for log in usage_logs if log.service_type == 'openai'])
            local_model_calls = len([log for log in usage_logs if log.service_type == 'local_model'])
            fallback_calls = len([log for log in usage_logs if log.service_type == 'fallback'])
            
            total_tokens = sum(log.total_tokens for log in usage_logs)
            total_cost = sum(log.estimated_cost for log in usage_logs)
            
            # Get recent usage (last 20 entries)
            recent_usage = session.query(ApiUsageLog).filter(
                ApiUsageLog.user_id == user_id
            ).order_by(ApiUsageLog.timestamp.desc()).limit(20).all()
            
            session.close()
            
            stats = {
                'total_calls': total_calls,
                'openai_calls': openai_calls,
                'local_model_calls': local_model_calls,
                'fallback_calls': fallback_calls,
                'total_tokens': total_tokens,
                'total_cost': round(total_cost, 6),
                'recent_usage': recent_usage,
                'period_days': days
            }
            
            logger.info(f"Retrieved usage stats for user {user_id}: {total_calls} calls, ${total_cost}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage stats for user {user_id}: {str(e)}")
            return {
                'total_calls': 0,
                'openai_calls': 0,
                'local_model_calls': 0,
                'fallback_calls': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'recent_usage': [],
                'period_days': days,
                'error': str(e)
            }
    
    def get_all_usage_stats(self, days: int = 30) -> Dict:
        """
        Get usage statistics for all users.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary containing aggregate usage statistics
        """
        try:
            session = self.Session()
            
            # Import here to avoid circular imports
            from app.models import ApiUsageLog, User
            from sqlalchemy import func
            
            # Calculate date range
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = start_date.replace(day=start_date.day - days)
            
            # Query aggregate statistics
            total_calls = session.query(func.count(ApiUsageLog.id)).filter(
                ApiUsageLog.timestamp >= start_date
            ).scalar()
            
            total_tokens = session.query(func.sum(ApiUsageLog.total_tokens)).filter(
                ApiUsageLog.timestamp >= start_date
            ).scalar() or 0
            
            total_cost = session.query(func.sum(ApiUsageLog.estimated_cost)).filter(
                ApiUsageLog.timestamp >= start_date
            ).scalar() or 0.0
            
            # Get usage by service type
            service_stats = session.query(
                ApiUsageLog.service_type,
                func.count(ApiUsageLog.id).label('count'),
                func.sum(ApiUsageLog.total_tokens).label('tokens'),
                func.sum(ApiUsageLog.estimated_cost).label('cost')
            ).filter(
                ApiUsageLog.timestamp >= start_date
            ).group_by(ApiUsageLog.service_type).all()
            
            # Get top users by usage
            top_users = session.query(
                User.username,
                func.count(ApiUsageLog.id).label('call_count'),
                func.sum(ApiUsageLog.total_tokens).label('total_tokens'),
                func.sum(ApiUsageLog.estimated_cost).label('total_cost')
            ).join(ApiUsageLog).filter(
                ApiUsageLog.timestamp >= start_date
            ).group_by(User.id, User.username).order_by(
                func.count(ApiUsageLog.id).desc()
            ).limit(10).all()
            
            session.close()
            
            stats = {
                'total_calls': total_calls,
                'total_tokens': total_tokens,
                'total_cost': round(total_cost, 6),
                'service_breakdown': [
                    {
                        'service_type': stat.service_type,
                        'count': stat.count,
                        'tokens': stat.tokens or 0,
                        'cost': round(stat.cost or 0.0, 6)
                    }
                    for stat in service_stats
                ],
                'top_users': [
                    {
                        'username': user.username,
                        'call_count': user.call_count,
                        'total_tokens': user.total_tokens or 0,
                        'total_cost': round(user.total_cost or 0.0, 6)
                    }
                    for user in top_users
                ],
                'period_days': days
            }
            
            logger.info(f"Retrieved aggregate usage stats: {total_calls} calls, ${total_cost}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get aggregate usage stats: {str(e)}")
            return {
                'total_calls': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'service_breakdown': [],
                'top_users': [],
                'period_days': days,
                'error': str(e)
            }
    
    def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old usage logs to prevent database bloat.
        
        Args:
            days_to_keep: Number of days of logs to keep
            
        Returns:
            Number of logs deleted
        """
        try:
            session = self.Session()
            
            # Import here to avoid circular imports
            from app.models import ApiUsageLog
            
            # Calculate cutoff date
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
            
            # Delete old logs
            deleted_count = session.query(ApiUsageLog).filter(
                ApiUsageLog.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            session.close()
            
            logger.info(f"Cleaned up {deleted_count} old usage logs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {str(e)}")
            return 0
