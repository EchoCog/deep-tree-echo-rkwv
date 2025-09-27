"""
API Endpoints for Task 2.6 and 2.7 Enhanced Functionality
Exposes explanation generation and preference learning capabilities via REST API
"""

from flask import Flask, request, jsonify
import asyncio
from enhanced_cognitive_integration import EnhancedCognitiveProcessor
from explanation_generation import ExplanationRequest, ExplanationStyle, ExplanationLevel
from enhanced_preference_learning import CommunicationStyle, InteractionPattern

app = Flask(__name__)

# Global processor instance
processor = None

def initialize_processor():
    """Initialize the enhanced cognitive processor"""
    global processor
    try:
        from persistent_memory_foundation import PersistentMemorySystem
        memory_system = PersistentMemorySystem("/tmp/api_memory")
        processor = EnhancedCognitiveProcessor(memory_system)
    except Exception:
        processor = EnhancedCognitiveProcessor()

@app.route('/api/v1/explanation/generate', methods=['POST'])
def generate_explanation():
    """Generate explanation for reasoning process"""
    try:
        data = request.json
        
        # Extract reasoning data
        reasoning_data = data.get('reasoning_data', {})
        user_preferences = data.get('user_preferences', {})
        detail_level = data.get('detail_level', 'detailed')
        
        # Generate explanation
        result = processor.generate_reasoning_explanation(
            reasoning_data, user_preferences, detail_level
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/explanation/cognitive', methods=['POST'])
def generate_cognitive_explanation():
    """Generate explanation for cognitive processing result"""
    try:
        data = request.json
        processing_result = data.get('processing_result', {})
        session_id = data.get('session_id', 'default')
        
        result = processor.generate_cognitive_explanation(processing_result, session_id)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/preferences/learn', methods=['POST'])
def learn_preferences():
    """Learn user preferences from interaction"""
    try:
        data = request.json
        
        session_id = data.get('session_id', 'default')
        query = data.get('query', '')
        response = data.get('response', '')
        conversation_history = data.get('conversation_history', [])
        feedback = data.get('feedback')
        
        result = processor.learn_user_preferences(
            session_id, query, response, conversation_history, feedback
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/preferences/context/<session_id>', methods=['GET'])
def get_personalized_context(session_id):
    """Get personalized context for session"""
    try:
        context = processor.get_personalized_context(session_id)
        return jsonify({'success': True, 'context': context})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/preferences/insights/<session_id>', methods=['GET'])
def get_user_insights(session_id):
    """Get user profile insights"""
    try:
        insights = processor.get_user_profile_insights(session_id)
        return jsonify({'success': True, 'insights': insights})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/system/enhanced-status', methods=['GET'])
def get_enhanced_status():
    """Get enhanced system status including Task 2.6 and 2.7 components"""
    try:
        status = processor.get_enhanced_system_status()
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/cognitive/process-enhanced', methods=['POST'])
async def process_enhanced():
    """Process input with enhanced cognitive capabilities including Task 2.6 & 2.7"""
    try:
        data = request.json
        
        input_text = data.get('input_text', '')
        session_id = data.get('session_id', 'default')
        conversation_history = data.get('conversation_history', [])
        memory_state = data.get('memory_state', {
            'declarative': {},
            'procedural': {},
            'episodic': [],
            'intentional': {'goals': []}
        })
        
        # Process with enhanced capabilities
        result = await processor.process_input_enhanced(
            input_text, session_id, conversation_history, memory_state
        )
        
        # Optionally generate explanation
        if data.get('generate_explanation', False):
            explanation_result = processor.generate_reasoning_explanation({
                'query': input_text,
                'conclusion': result.get('response', ''),
                'overall_confidence': result.get('confidence', 0.8)
            })
            result['explanation'] = explanation_result
        
        # Optionally learn preferences
        if data.get('learn_preferences', False):
            learning_result = processor.learn_user_preferences(
                session_id, input_text, result.get('response', ''), 
                conversation_history, data.get('feedback')
            )
            result['preference_learning'] = learning_result
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/demo/task-2-6-2-7', methods=['GET'])
def demo_endpoints():
    """Demo endpoint showing Task 2.6 and 2.7 capabilities"""
    return jsonify({
        'message': 'Task 2.6 & 2.7 Enhanced Functionality API',
        'endpoints': {
            'explanation_generation': {
                'POST /api/v1/explanation/generate': 'Generate explanations for reasoning processes',
                'POST /api/v1/explanation/cognitive': 'Generate explanations for cognitive processing'
            },
            'preference_learning': {
                'POST /api/v1/preferences/learn': 'Learn user preferences from interactions',
                'GET /api/v1/preferences/context/<session_id>': 'Get personalized context',
                'GET /api/v1/preferences/insights/<session_id>': 'Get user profile insights'
            },
            'integrated_processing': {
                'POST /api/v1/cognitive/process-enhanced': 'Enhanced cognitive processing with Task 2.6 & 2.7',
                'GET /api/v1/system/enhanced-status': 'System status including new components'
            }
        },
        'features': {
            'task_2_6': 'Explanation Generation System - Human-readable explanations with multiple styles',
            'task_2_7': 'Enhanced User Preference Learning - Communication styles and interaction patterns',
            'integration': 'Seamless integration with existing cognitive architecture'
        }
    })

if __name__ == '__main__':
    initialize_processor()
    app.run(debug=True, host='0.0.0.0', port=5000)