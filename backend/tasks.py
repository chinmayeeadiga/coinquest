# tasks.py - Background Task Scheduler
import asyncio
import schedule
import threading
import time
from datetime import datetime, time as dt_time
import logging
from typing import Optional
from trader import paper_trader
from agent import ai_agent
import config

logger = logging.getLogger(__name__)

class TaskScheduler:
    """
    Background task scheduler for the AI trading agent
    Handles periodic market analysis and system maintenance
    """
    
    def __init__(self):
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
    def start(self):
        """Start the background scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
            
        self.is_running = True
        
        # Schedule tasks
        self._schedule_tasks()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self.scheduler_thread.start()
        
        logger.info("Background task scheduler started")

    
    def stop(self):
        """Stop the background scheduler"""
        self.is_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        logger.info("Background task scheduler stopped")
    def _run_scheduler(self):
        """Background loop to run scheduled tasks"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)  # Prevents 100% CPU usage

    def _schedule_tasks(self):
        """Set up the task schedule"""
        
        # Continuous market analysis - every 2 minutes for real-time feel
        schedule.every(2).minutes.do(self._run_market_analysis)
        
        # More frequent analysis during likely market hours (9 AM - 4 PM ET)
        schedule.every(1).minutes.do(self._run_quick_scan)
        
        # Market analysis during trading hours
        # Pre-market analysis (9:00 AM ET)
        schedule.every().monday.at("14:00").do(self._run_market_analysis)  # 9:00 AM ET = 14:00 UTC
        schedule.every().tuesday.at("14:00").do(self._run_market_analysis)
        schedule.every().wednesday.at("14:00").do(self._run_market_analysis)
        schedule.every().thursday.at("14:00").do(self._run_market_analysis)
        schedule.every().friday.at("14:00").do(self._run_market_analysis)
        
        # Mid-day analysis (12:00 PM ET)
        schedule.every().monday.at("17:00").do(self._run_market_analysis)  # 12:00 PM ET = 17:00 UTC
        schedule.every().tuesday.at("17:00").do(self._run_market_analysis)
        schedule.every().wednesday.at("17:00").do(self._run_market_analysis)
        schedule.every().thursday.at("17:00").do(self._run_market_analysis)
        schedule.every().friday.at("17:00").do(self._run_market_analysis)
        
        # System maintenance tasks
        schedule.every(30).minutes.do(self._cleanup_expired_proposals)
        schedule.every().day.at("01:00").do(self._daily_maintenance)
        
        # Force initial analysis after 30 seconds
        schedule.every(30).seconds.do(self._initial_analysis).tag('initial')
        
        logger.info("Scheduled tasks configured - Running every 2 minutes + market hours")
    
    def _initial_analysis(self):
        """Run initial analysis once, then remove this task"""
        logger.info("Running initial analysis...")
        self._run_market_analysis()
        # Remove the initial analysis task so it only runs once
        schedule.clear('initial')
        return schedule.CancelJob  # Cancel this specific job
    
    def _run_quick_scan(self):
        """Quick market scan without full analysis"""
        try:
            # Just check for any urgent market moves or update portfolio values
            # This keeps the system feeling "alive" without heavy computation
            logger.info("Quick market scan...")
            
            # Update portfolio values (this will be used by the real-time UI)
            account_info = paper_trader.get_account_info()
            if account_info.get('error'):
                logger.warning(f"Portfolio update issue: {account_info['error']}")
            
        except Exception as e:
            logger.error(f"Error in quick scan: {e}")
    
    # def _run_market_analysis(self):
    #     """Run the main market analysis cycle"""
    #     logger.info("🚀 Starting scheduled market analysis with notifications")
        
    #     try:
    #         # Create new event loop for this thread
    #         if self.loop is None or self.loop.is_closed():
    #             self.loop = asyncio.new_event_loop()
    #             asyncio.set_event_loop(self.loop)
            
    #         # Run the analysis
    #         result = self.loop.run_until_complete(ai_agent.run_full_analysis_cycle())
            
    #         if result['status'] == 'success':
    #             proposals_count = result.get('proposals_generated', 0)
    #             notification_sent = result.get('notification_sent', False)
                
    #             logger.info(f"✅ Market analysis completed: {proposals_count} proposals generated")
                
    #             if notification_sent:
    #                 logger.info("📧 Email notification sent successfully")
    #             else:
    #                 logger.warning("⚠️ Email notification failed or no proposals to send")
                
    #             if proposals_count > 0:
    #                 logger.info(f"🎯 {proposals_count} trade proposals awaiting user approval")
    #             else:
    #                 logger.info("📊 No high-confidence opportunities found in this cycle")
                    
    #         else:
    #             logger.warning(f"Market analysis completed with status: {result['status']} - {result.get('message', '')}")
                
    #     except Exception as e:
    #         logger.error(f"❌ Error in scheduled market analysis: {e}")
    #         # Continue running despite errors
    def _run_market_analysis(self):
        """Run the market analysis cycle synchronously, handling the async context internally."""
        logger.info("🚀 Starting scheduled market analysis with notifications")
        
        try:
            # Create a brand new event loop for this synchronous task's execution
            loop = asyncio.new_event_loop()
            # Set it as the current loop for the thread
            asyncio.set_event_loop(loop)
            
            # Run the full async analysis cycle and block until completion
            # This setup is much more robust for async calls within a scheduler thread
            result = loop.run_until_complete(ai_agent.run_full_analysis_cycle())
            
            # Clean up the loop
            loop.close()
            
            logger.info(f"AI Agent analysis completed: {result}")
            
        except Exception as e:
            logger.error(f"AI Agent Analysis Cycle failed: {e}")
            # Add a clear warning that data is likely failing
            logger.warning("Data fetching failed. Check internet connection and API keys/limits.")
    def _cleanup_expired_proposals(self):
        """Clean up expired proposals"""
        try:
            ai_agent.cleanup_expired_proposals()
            logger.info("Expired proposals cleanup completed")
        except Exception as e:
            logger.error(f"Error in proposal cleanup: {e}")
    
    def _daily_maintenance(self):
        """Daily system maintenance tasks"""
        try:
            logger.info("Running daily maintenance tasks")
            
            # Cleanup expired proposals
            self._cleanup_expired_proposals()
            
            # Log system status
            status = ai_agent.get_system_status()
            logger.info(f"System status: {status['pending_proposals']} pending proposals, {status['active_positions']} active positions")
            
            # Additional maintenance tasks can be added here
            # - Database optimization
            # - Log rotation
            # - Performance metrics collection
            
        except Exception as e:
            logger.error(f"Error in daily maintenance: {e}")
    
    def trigger_immediate_analysis(self):
        """Trigger an immediate market analysis (for manual testing)"""
        logger.info("Triggering immediate market analysis")
        threading.Thread(target=self._run_market_analysis, daemon=True).start()

# Global scheduler instance
task_scheduler = TaskScheduler()

# Convenience functions
def start_scheduler():
    """Start the background scheduler"""
    task_scheduler.start()

def stop_scheduler():
    """Stop the background scheduler"""
    task_scheduler.stop()

def trigger_analysis():
    """Trigger immediate analysis"""
    task_scheduler.trigger_immediate_analysis()

