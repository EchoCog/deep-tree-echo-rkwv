#!/usr/bin/env python3
"""
Deep Tree Echo CLI Tool
Command-line interface for managing the Deep Tree Echo cognitive architecture
"""

import click
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
import requests
from tabulate import tabulate

# Import SDK components
sys.path.append(str(Path(__file__).parent.parent / 'sdk' / 'python'))
from client import EchoClient
from exceptions import EchoAPIError, EchoAuthenticationError, EchoRateLimitError

class EchoCLI:
    """Main CLI class"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.client = EchoClient(api_key, base_url)
        self.base_url = base_url
    
    def format_output(self, data: Any, output_format: str = "table") -> str:
        """Format output data"""
        if output_format == "json":
            return json.dumps(data, indent=2, default=str)
        elif output_format == "table" and isinstance(data, list) and data:
            # Create table from list of dicts
            if isinstance(data[0], dict):
                headers = list(data[0].keys())
                rows = [[item.get(key, '') for key in headers] for item in data]
                return tabulate(rows, headers=headers, tablefmt="grid")
        elif output_format == "table" and isinstance(data, dict):
            # Create table from dict
            rows = [[key, value] for key, value in data.items()]
            return tabulate(rows, headers=["Key", "Value"], tablefmt="grid")
        
        return str(data)

@click.group()
@click.option('--api-key', envvar='ECHO_API_KEY', required=True, help='Echo API key')
@click.option('--base-url', envvar='ECHO_BASE_URL', default='http://localhost:8000', help='Base URL for Echo API')
@click.option('--output', '-o', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.pass_context
def cli(ctx, api_key, base_url, output):
    """Deep Tree Echo CLI - Manage your cognitive architecture from the command line"""
    ctx.ensure_object(dict)
    ctx.obj['cli'] = EchoCLI(api_key, base_url)
    ctx.obj['output_format'] = output

@cli.group()
def system():
    """System management commands"""
    pass

@system.command()
@click.pass_context
def status(ctx):
    """Get system status"""
    try:
        cli_obj = ctx.obj['cli']
        status_data = cli_obj.client.get_system_status()
        
        output = cli_obj.format_output(status_data.__dict__, ctx.obj['output_format'])
        click.echo(output)
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@system.command()
@click.pass_context
def health(ctx):
    """Check system health"""
    try:
        cli_obj = ctx.obj['cli']
        is_healthy = cli_obj.client.health_check()
        
        if is_healthy:
            click.echo("✅ System is healthy")
        else:
            click.echo("❌ System is not healthy")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error checking health: {e}", err=True)
        sys.exit(1)

@system.command()
@click.pass_context
def ping(ctx):
    """Ping the API server"""
    try:
        cli_obj = ctx.obj['cli']
        response_time = cli_obj.client.ping()
        click.echo(f"🏓 Pong! Response time: {response_time}ms")
        
    except Exception as e:
        click.echo(f"Error pinging server: {e}", err=True)
        sys.exit(1)

@cli.group()
def cognitive():
    """Cognitive processing commands"""
    pass

@cognitive.command()
@click.argument('input_text')
@click.option('--session-id', help='Session ID for context')
@click.option('--temperature', type=float, default=0.8, help='Temperature parameter (0.0-2.0)')
@click.option('--max-tokens', type=int, default=2048, help='Maximum tokens to generate')
@click.pass_context
def process(ctx, input_text, session_id, temperature, max_tokens):
    """Process cognitive input"""
    try:
        cli_obj = ctx.obj['cli']
        result = cli_obj.client.process_cognitive_input(
            input_text,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if ctx.obj['output_format'] == 'json':
            output = cli_obj.format_output(result.__dict__, 'json')
        else:
            # Pretty format for cognitive results
            click.echo(f"🧠 Input: {result.input_text}")
            click.echo(f"💭 Response: {result.integrated_response}")
            click.echo(f"⏱️  Processing Time: {result.processing_time:.3f}s")
            click.echo(f"🔗 Session ID: {result.session_id}")
            if result.confidence:
                click.echo(f"📊 Confidence: {result.confidence:.2%}")
            return
        
        click.echo(output)
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cognitive.command()
@click.argument('inputs', nargs=-1, required=True)
@click.option('--session-id', help='Session ID for context')
@click.option('--temperature', type=float, default=0.8, help='Temperature parameter')
@click.option('--concurrency', type=int, default=5, help='Number of concurrent requests')
@click.pass_context
def batch(ctx, inputs, session_id, temperature, concurrency):
    """Process multiple inputs in batch"""
    try:
        cli_obj = ctx.obj['cli']
        
        click.echo(f"Processing {len(inputs)} inputs with concurrency {concurrency}...")
        
        # Use asyncio for batch processing
        async def run_batch():
            return cli_obj.client.batch_process(
                list(inputs),
                session_id=session_id,
                temperature=temperature,
                concurrency=concurrency
            )
        
        results = asyncio.run(run_batch())
        
        if ctx.obj['output_format'] == 'json':
            output = cli_obj.format_output([r.__dict__ for r in results], 'json')
            click.echo(output)
        else:
            for i, result in enumerate(results, 1):
                click.echo(f"\n--- Result {i} ---")
                click.echo(f"Input: {result.input_text}")
                click.echo(f"Response: {result.integrated_response}")
                click.echo(f"Time: {result.processing_time:.3f}s")
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.group()
def session():
    """Session management commands"""
    pass

@session.command()
@click.option('--temperature', type=float, default=0.8, help='Temperature parameter')
@click.option('--max-context', type=int, default=2048, help='Maximum context length')
@click.option('--memory-persistence/--no-memory-persistence', default=True, help='Enable memory persistence')
@click.pass_context
def create(ctx, temperature, max_context, memory_persistence):
    """Create a new session"""
    try:
        cli_obj = ctx.obj['cli']
        
        from models import SessionConfiguration
        config = SessionConfiguration(
            temperature=temperature,
            max_context_length=max_context,
            memory_persistence=memory_persistence
        )
        
        session_info = cli_obj.client.create_session(config)
        
        if ctx.obj['output_format'] == 'json':
            output = cli_obj.format_output(session_info.__dict__, 'json')
        else:
            click.echo(f"✅ Session created successfully!")
            click.echo(f"Session ID: {session_info.session_id}")
            click.echo(f"Status: {session_info.status}")
            click.echo(f"Created: {session_info.created_at}")
            return
        
        click.echo(output)
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@session.command()
@click.argument('session_id')
@click.pass_context
def info(ctx, session_id):
    """Get session information"""
    try:
        cli_obj = ctx.obj['cli']
        session_info = cli_obj.client.get_session(session_id)
        
        output = cli_obj.format_output(session_info.__dict__, ctx.obj['output_format'])
        click.echo(output)
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.group()
def memory():
    """Memory management commands"""
    pass

@memory.command()
@click.argument('query')
@click.option('--type', 'memory_type', help='Memory type filter (declarative, procedural, episodic)')
@click.option('--limit', type=int, default=10, help='Maximum number of results')
@click.option('--min-relevance', type=float, default=0.5, help='Minimum relevance score')
@click.pass_context
def search(ctx, query, memory_type, limit, min_relevance):
    """Search memory items"""
    try:
        cli_obj = ctx.obj['cli']
        results = cli_obj.client.search_memory(
            query,
            memory_type=memory_type,
            limit=limit,
            min_relevance=min_relevance
        )
        
        if ctx.obj['output_format'] == 'json':
            output = cli_obj.format_output([r.__dict__ for r in results], 'json')
        else:
            if not results:
                click.echo("No memory items found matching the query.")
                return
                
            click.echo(f"Found {len(results)} memory items:")
            for i, item in enumerate(results, 1):
                click.echo(f"\n--- Item {i} ---")
                click.echo(f"ID: {item.id}")
                click.echo(f"Type: {item.memory_type}")
                click.echo(f"Content: {item.content[:100]}{'...' if len(item.content) > 100 else ''}")
                click.echo(f"Relevance: {item.relevance_score:.2%}")
                click.echo(f"Access Count: {item.access_count}")
            return
        
        click.echo(output)
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.group()
def analytics():
    """Analytics and usage commands"""
    pass

@analytics.command()
@click.option('--period', default='last_30_days', help='Analytics period')
@click.pass_context
def usage(ctx, period):
    """Get usage analytics"""
    try:
        cli_obj = ctx.obj['cli']
        analytics = cli_obj.client.get_usage_analytics(period)
        
        if ctx.obj['output_format'] == 'json':
            output = cli_obj.format_output(analytics.__dict__, 'json')
        else:
            click.echo(f"📊 Usage Analytics ({period})")
            click.echo(f"Total Requests: {analytics.total_requests:,}")
            click.echo(f"Successful: {analytics.successful_requests:,}")
            click.echo(f"Errors: {analytics.error_requests:,}")
            click.echo(f"Success Rate: {analytics.successful_requests/analytics.total_requests:.1%}")
            click.echo(f"Avg Response Time: {analytics.average_response_time:.3f}s")
            click.echo(f"API Tier: {analytics.api_tier}")
            click.echo(f"Quota Remaining: {analytics.quota_remaining:,}")
            return
        
        click.echo(output)
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@analytics.command()
@click.pass_context
def quota(ctx):
    """Check API quota status"""
    try:
        cli_obj = ctx.obj['cli']
        
        # Make a simple request to get quota headers
        cli_obj.client.health_check()
        quota_info = cli_obj.client.quota_info
        
        if quota_info:
            if ctx.obj['output_format'] == 'json':
                output = cli_obj.format_output(quota_info.__dict__, 'json')
                click.echo(output)
            else:
                click.echo(f"📈 API Quota Status")
                click.echo(f"Tier: {quota_info.tier}")
                click.echo(f"Hour Usage: {quota_info.hour_usage}/{quota_info.hour_limit}")
                click.echo(f"Day Usage: {quota_info.day_usage}/{quota_info.day_limit}")
                click.echo(f"Status: {'✅ OK' if quota_info.allowed else '❌ EXCEEDED'}")
        else:
            click.echo("No quota information available")
        
    except EchoAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.group()
def config():
    """Configuration management commands"""
    pass

@config.command()
@click.option('--api-key', help='Set API key')
@click.option('--base-url', help='Set base URL')
def set(api_key, base_url):
    """Set configuration values"""
    config_file = Path.home() / '.echo' / 'config.json'
    config_file.parent.mkdir(exist_ok=True)
    
    config_data = {}
    if config_file.exists():
        config_data = json.loads(config_file.read_text())
    
    if api_key:
        config_data['api_key'] = api_key
        click.echo("✅ API key updated")
    
    if base_url:
        config_data['base_url'] = base_url
        click.echo("✅ Base URL updated")
    
    config_file.write_text(json.dumps(config_data, indent=2))
    click.echo(f"Configuration saved to {config_file}")

@config.command()
def show():
    """Show current configuration"""
    config_file = Path.home() / '.echo' / 'config.json'
    
    if config_file.exists():
        config_data = json.loads(config_file.read_text())
        # Hide API key for security
        if 'api_key' in config_data:
            config_data['api_key'] = config_data['api_key'][:10] + '...'
        
        click.echo(json.dumps(config_data, indent=2))
    else:
        click.echo("No configuration file found")

@cli.command()
@click.option('--count', type=int, default=1, help='Number of test requests')
@click.pass_context
def test(ctx, count):
    """Run API tests"""
    cli_obj = ctx.obj['cli']
    
    click.echo(f"🧪 Running {count} test request(s)...")
    
    success_count = 0
    total_time = 0
    
    for i in range(count):
        try:
            start_time = datetime.now()
            
            # Test cognitive processing
            result = cli_obj.client.process_cognitive_input(f"Test input {i+1}")
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            total_time += response_time
            
            success_count += 1
            click.echo(f"  ✅ Test {i+1}: {response_time:.3f}s")
            
        except Exception as e:
            click.echo(f"  ❌ Test {i+1}: {e}")
    
    click.echo(f"\n📊 Test Results:")
    click.echo(f"  Success Rate: {success_count}/{count} ({success_count/count:.1%})")
    click.echo(f"  Average Response Time: {total_time/count:.3f}s")
    
    if success_count < count:
        sys.exit(1)

if __name__ == '__main__':
    # Install required packages if not available
    try:
        import tabulate
    except ImportError:
        click.echo("Installing required dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tabulate'])
        import tabulate
    
    cli()