# models.py - FIXED Database Manager - Allows Duplicates & Works with React Frontend
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import threading

logger = logging.getLogger(__name__)

class DatabaseManager:
    """FIXED: Database manager that allows duplicates and works reliably"""
    
    def __init__(self, db_file: str = 'new_db.db'):
        self.db_file = db_file
        self.lock = threading.Lock()
        self._init_database()
        
        # FIXED: Allow duplicates and reduce restrictions
        self.proposal_cooldown = 30  # REDUCED to 30 seconds
        self.max_pending_per_symbol = 10  # INCREASED to allow multiple
        
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # FIXED: Removed UNIQUE constraint to allow duplicates
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS proposals (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        current_price REAL,
                        target_price REAL,
                        stop_loss REAL,
                        position_size REAL,
                        reasoning TEXT,
                        technical_score REAL,
                        ml_score REAL,
                        risk_score REAL,
                        volatility_forecast REAL,
                        model_predictions TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        execution_result TEXT,
                        user_action TEXT
                    )
                ''')
                
                # Add indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_status ON proposals(symbol, status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON proposals(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON proposals(status)')
                
                # Trade history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id TEXT PRIMARY KEY,
                        proposal_id TEXT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        status TEXT NOT NULL,
                        exchange TEXT,
                        alpaca_order_id TEXT,
                        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        notes TEXT,
                        FOREIGN KEY(proposal_id) REFERENCES proposals(id)
                    )
                ''')
                
                # Portfolio snapshots table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_value REAL,
                        cash REAL,
                        positions TEXT,
                        daily_pnl REAL,
                        total_pnl REAL,
                        snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully - duplicates allowed")
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def save_agent_proposal(self, proposal_data: Dict[str, Any]) -> Optional[str]:
        """FIXED: Save proposal - allows duplicates and is more permissive"""
        with self.lock:
            try:
                symbol = proposal_data.get('symbol', '').upper()
                action = proposal_data.get('action', '')
                
                if not symbol or not action:
                    logger.error("Invalid proposal data: missing symbol or action")
                    return None
                
                # FIXED: Much more permissive duplicate checking
                if self._has_too_many_pending(symbol):
                    logger.info(f"Too many pending proposals for {symbol}, but allowing anyway")
                    # Don't return None - continue with save
                
                proposal_id = str(uuid.uuid4())
                expires_at = datetime.now() + timedelta(hours=2)
                
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    
                    # Clean up old expired proposals first
                    cursor.execute('''
                        DELETE FROM proposals 
                        WHERE status = 'pending' AND expires_at < ?
                    ''', (datetime.now().isoformat(),))
                    
                    # Insert new proposal
                    cursor.execute('''
                        INSERT INTO proposals (
                            id, symbol, action, confidence, current_price, target_price,
                            stop_loss, position_size, reasoning, technical_score, ml_score,
                            risk_score, volatility_forecast, model_predictions, expires_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        proposal_id,
                        symbol,
                        action,
                        proposal_data.get('confidence', 0.5),
                        proposal_data.get('current_price'),
                        proposal_data.get('target_price'),
                        proposal_data.get('stop_loss'),
                        proposal_data.get('position_size', 0.02),
                        proposal_data.get('reasoning', ''),
                        proposal_data.get('technical_score', 0.5),
                        proposal_data.get('ml_score', 0.5),
                        proposal_data.get('risk_score', 0.3),
                        proposal_data.get('volatility_forecast', 0.02),
                        json.dumps(proposal_data.get('model_predictions', {})),
                        expires_at.isoformat()
                    ))
                    
                    conn.commit()
                    logger.info(f"✅ Successfully saved proposal {proposal_id} for {symbol} ({action})")
                    return proposal_id
                    
            except Exception as e:
                logger.error(f"Error saving proposal: {e}")
                return None
    
    def _has_too_many_pending(self, symbol: str) -> bool:
        """Check if symbol has too many pending proposals"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM proposals 
                    WHERE symbol = ? AND status = 'pending' AND expires_at > ?
                ''', (symbol.upper(), datetime.now().isoformat()))
                
                count = cursor.fetchone()[0]
                return count >= self.max_pending_per_symbol  # Now allows up to 10
                
        except Exception as e:
            logger.error(f"Error checking pending proposals: {e}")
            return False
    
    def get_pending_proposals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending proposals with automatic cleanup"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Clean up expired first
                cursor.execute('''
                    UPDATE proposals 
                    SET status = 'expired', updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'pending' AND expires_at < ?
                ''', (datetime.now().isoformat(),))
                
                # Get pending proposals
                cursor.execute('''
                    SELECT * FROM proposals 
                    WHERE status = 'pending' AND expires_at > ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (datetime.now().isoformat(), limit))
                
                proposals = []
                for row in cursor.fetchall():
                    proposal = dict(row)
                    
                    # Parse JSON fields
                    if proposal.get('model_predictions'):
                        try:
                            proposal['model_predictions'] = json.loads(proposal['model_predictions'])
                        except:
                            proposal['model_predictions'] = {}
                    
                    # Calculate time remaining
                    if proposal.get('expires_at'):
                        try:
                            expires_at = datetime.fromisoformat(proposal['expires_at'])
                            proposal['time_remaining_minutes'] = max(0, 
                                int((expires_at - datetime.now()).total_seconds() / 60))
                        except:
                            proposal['time_remaining_minutes'] = 120
                    
                    proposals.append(proposal)
                
                logger.info(f"Retrieved {len(proposals)} pending proposals from database")
                return proposals
                
        except Exception as e:
            logger.error(f"Error getting pending proposals: {e}")
            return []
    
    def update_proposal_status(self, proposal_id: str, status: str, 
                             execution_result: Optional[Dict] = None) -> bool:
        """Update proposal status"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                update_data = [status, datetime.now().isoformat()]
                query = '''
                    UPDATE proposals 
                    SET status = ?, updated_at = ?
                '''
                
                if execution_result:
                    query += ', execution_result = ?'
                    update_data.append(json.dumps(execution_result))
                
                query += ' WHERE id = ?'
                update_data.append(proposal_id)
                
                cursor.execute(query, tuple(update_data))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Updated proposal {proposal_id} status to {status}")
                    return True
                else:
                    logger.warning(f"Proposal {proposal_id} not found for update")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating proposal status: {e}")
            return False
    
    def get_proposal_by_id(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get specific proposal by ID"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM proposals WHERE id = ?', (proposal_id,))
                row = cursor.fetchone()
                
                if row:
                    proposal = dict(row)
                    if proposal.get('model_predictions'):
                        try:
                            proposal['model_predictions'] = json.loads(proposal['model_predictions'])
                        except:
                            proposal['model_predictions'] = {}
                    return proposal
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting proposal {proposal_id}: {e}")
            return None
    
    def save_trade_execution(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """Save trade execution record"""
        try:
            trade_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades (
                        id, proposal_id, symbol, side, quantity, price,
                        status, exchange, alpaca_order_id, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    trade_data.get('proposal_id'),
                    trade_data.get('symbol'),
                    trade_data.get('side'),
                    trade_data.get('quantity'),
                    trade_data.get('price'),
                    trade_data.get('status'),
                    trade_data.get('exchange'),
                    trade_data.get('alpaca_order_id'),
                    trade_data.get('notes', '')
                ))
                
                conn.commit()
                logger.info(f"✅ Saved trade execution {trade_id}")
                return trade_id
                
        except Exception as e:
            logger.error(f"Error saving trade execution: {e}")
            return None
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade execution history"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades 
                    ORDER BY executed_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def clear_all_proposals(self) -> bool:
        """Clear all proposals (for testing)"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM proposals')
                conn.commit()
                logger.info("Cleared all proposals from database")
                return True
        except Exception as e:
            logger.error(f"Error clearing proposals: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Proposal stats
                cursor.execute('SELECT status, COUNT(*) FROM proposals GROUP BY status')
                proposal_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Trade stats
                cursor.execute('SELECT side, COUNT(*) FROM trades GROUP BY side')
                trade_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Recent activity
                cursor.execute('''
                    SELECT COUNT(*) FROM proposals 
                    WHERE created_at > datetime('now', '-24 hours')
                ''')
                proposals_24h = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT COUNT(*) FROM trades 
                    WHERE executed_at > datetime('now', '-24 hours')
                ''')
                trades_24h = cursor.fetchone()[0]
                
                return {
                    'proposal_stats': proposal_stats,
                    'trade_stats': trade_stats,
                    'proposals_24h': proposals_24h,
                    'trades_24h': trades_24h,
                    'database_file': self.db_file,
                    'duplicate_prevention': 'disabled',
                    'max_pending_per_symbol': self.max_pending_per_symbol,
                    'proposal_cooldown_seconds': self.proposal_cooldown,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def cleanup_expired_proposals(self):
        """Clean up expired proposals"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE proposals 
                    SET status = 'expired', updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'pending' AND expires_at < ?
                ''', (datetime.now().isoformat(),))
                
                if cursor.rowcount > 0:
                    logger.info(f"Cleaned up {cursor.rowcount} expired proposals")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired proposals: {e}")

# Global instance
db_manager = DatabaseManager()

# Test function to verify database works
def test_database():
    """Test database functionality"""
    try:
        logger.info("Testing database functionality...")
        
        # Test proposal creation
        test_proposal = {
            'symbol': 'TEST',
            'action': 'BUY',
            'confidence': 0.75,
            'current_price': 100.0,
            'target_price': 110.0,
            'reasoning': 'Database test proposal'
        }
        
        proposal_id = db_manager.save_agent_proposal(test_proposal)
        if proposal_id:
            logger.info(f"✅ Database test passed - created proposal {proposal_id}")
            
            # Test retrieval
            proposals = db_manager.get_pending_proposals()
            logger.info(f"✅ Database test passed - retrieved {len(proposals)} proposals")
            
            # Clean up test
            db_manager.update_proposal_status(proposal_id, 'test_cleanup')
            
            return True
        else:
            logger.error("❌ Database test failed - could not create proposal")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    test_database()