# notifier.py - FIXED Email Notification Service
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# FIXED: Base URL for email approval links
BASE_URL = "https://coin-pilot-3.onrender.com"  # Your FastAPI server

class EmailNotifier:
    """FIXED: Email notification service with proper Gmail configuration"""
    
    def __init__(self):
        # FIXED: Correct Gmail SMTP settings
        self.smtp_server = config.EMAIL_HOST  # smtp.gmail.com
        self.port = config.EMAIL_PORT  # 587 (NOT 465)
        self.sender_email = config.EMAIL_ADDRESS
        self.password = config.EMAIL_PASSWORD  # App Password
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate email configuration"""
        if not all([self.smtp_server, self.sender_email, self.password]):
            logger.error("Email configuration incomplete")
            return False
            
        if self.port != 587:
            logger.error(f"Gmail requires port 587, not {self.port}")
            return False
            
        if len(self.password) < 16:
            logger.warning("Gmail App Password should be 16 characters")
            
        return True
        
    def send_trade_proposals(self, proposals: List[Dict], recipient_email: str) -> bool:
        """FIXED: Send email with trade proposals - proper Gmail STARTTLS"""
        try:
            logger.info(f"Sending {len(proposals)} trade proposals to {recipient_email}")
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"AI Trading Agent - {len(proposals)} Trade Proposals ({datetime.now().strftime('%H:%M')})"
            message["From"] = self.sender_email
            message["To"] = recipient_email
            
            # Create HTML and text content
            html_content = self._create_proposal_email_html(proposals)
            text_content = self._create_proposal_email_text(proposals)
            
            message.attach(MIMEText(text_content, "plain"))
            message.attach(MIMEText(html_content, "html"))
            
            # FIXED: Proper Gmail SMTP with STARTTLS
            context = ssl.create_default_context()
            
            logger.info(f"Connecting to {self.smtp_server}:{self.port}...")
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                # FIXED: Use STARTTLS (not SSL) for port 587
                logger.info("Starting TLS...")
                server.starttls(context=context)
                
                logger.info(f"Logging in as {self.sender_email}...")
                server.login(self.sender_email, self.password)
                
                logger.info(f"Sending email...")
                server.sendmail(self.sender_email, recipient_email, message.as_string())
            
            logger.info(f"Successfully sent {len(proposals)} trade proposals")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Gmail authentication failed: {e}")
            logger.error("Check: 1) EMAIL_ADDRESS correct 2) EMAIL_PASSWORD is 16-char App Password 3) 2FA enabled")
            return False
        except smtplib.SMTPServerDisconnected as e:
            logger.error(f"SMTP server disconnected: {e}")
            logger.error("Gmail may have blocked the connection. Check security settings.")
            return False
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False
    
    def _create_proposal_email_html(self, proposals: List[Dict]) -> str:
        """Create modern HTML email with trade proposals"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Trading Proposals</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; }}
                .proposal {{ border: 1px solid #e0e0e0; margin: 20px 0; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .proposal.buy {{ border-left: 5px solid #4CAF50; background: #f8fff8; }}
                .proposal.sell {{ border-left: 5px solid #f44336; background: #fff8f8; }}
                .crypto-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }}
                .crypto-symbol {{ font-size: 1.5em; font-weight: bold; }}
                .action-badge {{ padding: 8px 16px; border-radius: 20px; font-weight: bold; color: white; }}
                .action-buy {{ background: #4CAF50; }}
                .action-sell {{ background: #f44336; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                .metric-value {{ font-size: 1.3em; font-weight: bold; color: #2196F3; }}
                .metric-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
                .reasoning {{ background: #f0f4f8; padding: 15px; border-radius: 8px; margin: 15px 0; font-style: italic; }}
                .actions {{ text-align: center; margin: 25px 0; }}
                .btn {{ display: inline-block; padding: 12px 30px; margin: 0 8px; text-decoration: none; border-radius: 25px; font-weight: bold; transition: all 0.3s; }}
                .btn-approve {{ background: #4CAF50; color: white; }}
                .btn-approve:hover {{ background: #45a049; }}
                .btn-reject {{ background: #f44336; color: white; }}
                .btn-reject:hover {{ background: #da190b; }}
                .dashboard-link {{ background: linear-gradient(135deg, #2196F3, #21cbf3); color: white; padding: 20px; text-align: center; border-radius: 10px; margin: 30px 0; }}
                .footer {{ margin-top: 40px; padding: 20px; border-top: 2px solid #eee; font-size: 0.9em; color: #666; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat {{ text-align: center; }}
                .stat-number {{ font-size: 1.5em; font-weight: bold; color: #2196F3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AI Crypto Trading Agent</h1>
                    <p>Real-time market analysis has identified <strong>{len(proposals)}</strong> high-confidence trading opportunities</p>
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-number">{len([p for p in proposals if p['action'] == 'BUY'])}</div>
                            <div>BUY Signals</div>
                        </div>
                        <div class="stat">
                            <div class="stat-number">{len([p for p in proposals if p['action'] == 'SELL'])}</div>
                            <div>SELL Signals</div>
                        </div>
                        <div class="stat">
                            <div class="stat-number">{datetime.now().strftime('%H:%M')}</div>
                            <div>Generated</div>
                        </div>
                    </div>
                </div>
                
                <div class="dashboard-link">
                    <h3>Quick Access</h3>
                    <a href="{BASE_URL}/dashboard" style="color: white; text-decoration: none; font-size: 1.1em;">
                        Open Web Dashboard for Bulk Actions & Live Portfolio
                    </a>
                </div>
        """
        
        # Add each proposal
        for i, proposal in enumerate(proposals, 1):
            proposal_id = proposal['id']
            symbol = proposal['symbol']
            action = proposal['action'].upper()
            confidence = proposal['confidence']
            current_price = proposal.get('current_price', 0)
            target_price = proposal.get('target_price', 0)
            reasoning = proposal.get('reasoning', 'AI analysis indicates trading opportunity')
            
            # Calculate potential return
            potential_return = 0
            if target_price and current_price and current_price > 0:
                if action == 'BUY':
                    potential_return = ((target_price - current_price) / current_price) * 100
                else:
                    potential_return = ((current_price - target_price) / current_price) * 100
            
            action_class = action.lower()
            confidence_color = "#4CAF50" if confidence > 0.7 else "#ff9800" if confidence > 0.5 else "#f44336"
            
            html += f"""
                <div class="proposal {action_class}">
                    <div class="crypto-header">
                        <div class="crypto-symbol">{symbol}</div>
                        <div class="action-badge action-{action_class}">{action}</div>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value" style="color: {confidence_color}">{confidence:.1%}</div>
                            <div class="metric-label">AI Confidence</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${current_price:,.6f}</div>
                            <div class="metric-label">Current Price</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${target_price:,.6f}</div>
                            <div class="metric-label">Target Price</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: {'#4CAF50' if potential_return > 0 else '#f44336'}">{potential_return:+.1f}%</div>
                            <div class="metric-label">Expected Return</div>
                        </div>
                    </div>
                    
                    <div class="reasoning">
                        <strong>AI Analysis:</strong> {reasoning[:200]}{"..." if len(reasoning) > 200 else ""}
                    </div>
                    
                    <div class="actions">
                        <a href="{BASE_URL}/approve/{proposal_id}" class="btn btn-approve">
                            APPROVE & EXECUTE
                        </a>
                        <a href="{BASE_URL}/reject/{proposal_id}" class="btn btn-reject">
                            REJECT
                        </a>
                    </div>
                </div>
            """
        
        html += f"""
                <div class="footer">
                    <h3>Important Information</h3>
                    <ul>
                        <li><strong>Real-time Analysis:</strong> All proposals based on live market data and AI analysis</li>
                        <li><strong>Paper Trading:</strong> Trades execute in simulation mode - no real money at risk</li>
                        <li><strong>Response Time:</strong> Proposals expire in 2 hours for security</li>
                        <li><strong>Platform:</strong> Using Alpaca Paper Trading + Real-time crypto data</li>
                        <li><strong>Coverage:</strong> Supports {len(config.ALL_CRYPTO_SYMBOLS)} cryptocurrencies</li>
                    </ul>
                    
                    <div style="margin-top: 20px; padding: 15px; background: #e8f4f8; border-radius: 8px;">
                        <strong>System Status:</strong> AI Trading Agent v2.0 | Real-time Data Active | Paper Trading Mode
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_proposal_email_text(self, proposals: List[Dict]) -> str:
        """Create plain text email content"""
        
        text = f"""
        AI CRYPTO TRADING AGENT - TRADE PROPOSALS
        =========================================
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Total Proposals: {len(proposals)}
        BUY Signals: {len([p for p in proposals if p['action'] == 'BUY'])}
        SELL Signals: {len([p for p in proposals if p['action'] == 'SELL'])}
        
        Web Dashboard: {BASE_URL}/dashboard
        
        """
        
        for i, proposal in enumerate(proposals, 1):
            symbol = proposal['symbol']
            action = proposal['action'].upper()
            confidence = proposal['confidence']
            current_price = proposal.get('current_price', 0)
            target_price = proposal.get('target_price', 0)
            reasoning = proposal.get('reasoning', 'AI trading opportunity')
            proposal_id = proposal['id']
            
            potential_return = 0
            if target_price and current_price and current_price > 0:
                if action == 'BUY':
                    potential_return = ((target_price - current_price) / current_price) * 100
                else:
                    potential_return = ((current_price - target_price) / current_price) * 100
            
            text += f"""
        PROPOSAL #{i}: {symbol} - {action}
        --------------------------------
        Confidence: {confidence:.1%}
        Current Price: ${current_price:,.6f}
        Target Price: ${target_price:,.6f}
        Expected Return: {potential_return:+.1f}%
        
        AI Analysis: {reasoning[:150]}{"..." if len(reasoning) > 150 else ""}
        
        Actions:
        - APPROVE: {BASE_URL}/approve/{proposal_id}
        - REJECT:  {BASE_URL}/reject/{proposal_id}
        
        """
        
        text += f"""
        
        SYSTEM INFORMATION:
        ==================
        - Real-time market data from multiple exchanges
        - Paper trading mode (no real money at risk)
        - Supports {len(config.ALL_CRYPTO_SYMBOLS)} cryptocurrencies
        - Alpaca integration for supported pairs
        - Proposals expire in 2 hours
        
        AI Trading Agent v2.0 | Paper Trading Mode Active
        """
        
        return text
    
    def send_execution_confirmation(self, proposal: Dict, action: str, 
                                  result: Dict, recipient_email: str) -> bool:
        """Send confirmation email after trade execution"""
        try:
            success = result.get('status') == 'success'
            subject = f"Trade {'Executed' if success else 'Failed'}: {proposal['symbol']} - {action.upper()}"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="background: {'#4CAF50' if success else '#f44336'}; color: white; padding: 25px; border-radius: 10px; text-align: center;">
                        <h1>{'Trade Executed Successfully' if success else 'Trade Execution Failed'}</h1>
                        <h2>{proposal['symbol']} - {action.upper()}</h2>
                        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div style="padding: 25px; border: 1px solid #ddd; margin: 20px 0; border-radius: 10px;">
                        <h3>Execution Details</h3>
                        <p><strong>Status:</strong> {result.get('status', 'unknown').upper()}</p>
                        <p><strong>Message:</strong> {result.get('message', 'No details available')}</p>
                        <p><strong>Platform:</strong> {result.get('trading_platform', 'Unknown')}</p>
                        
                        {f"<p><strong>P&L:</strong> ${result.get('pnl', 0):+.2f} ({result.get('pnl_percentage', 0):+.1f}%)</p>" if 'pnl' in result else ""}
                        
                        <h3>Original Proposal</h3>
                        <p><strong>Symbol:</strong> {proposal['symbol']}</p>
                        <p><strong>Action:</strong> {proposal['action'].upper()}</p>
                        <p><strong>Price:</strong> ${proposal.get('current_price', 0):,.6f}</p>
                        <p><strong>Confidence:</strong> {proposal.get('confidence', 0):.1%}</p>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{BASE_URL}/dashboard" 
                           style="background-color: #2196F3; color: white; padding: 15px 25px; text-decoration: none; border-radius: 5px; font-weight: bold;">
                            View Portfolio Dashboard
                        </a>
                    </div>
                </div>
            </body>
            </html>
            """
            
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient_email
            
            message.attach(MIMEText(html_content, "html"))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, recipient_email, message.as_string())
            
            logger.info(f"Sent execution confirmation for {proposal['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send execution confirmation: {e}")
            return False
    
    def test_email_connection(self) -> bool:
        """Test email connection and configuration"""
        try:
            logger.info("Testing Gmail connection...")
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.password)
                logger.info("Gmail connection test successful!")
                return True
                
        except smtplib.SMTPAuthenticationError:
            logger.error("Gmail authentication failed - check App Password")
            return False
        except Exception as e:
            logger.error(f"Email test failed: {e}")
            return False

# Global instance
email_notifier = EmailNotifier()