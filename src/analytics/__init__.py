"""
Advanced Analytics and Reporting System for Deep Tree Echo
Implements comprehensive analytics, reporting, and business intelligence capabilities.
"""

try:
    from .data_models import (
        AnalyticsEvent,
        UserSession,
        SystemMetrics,
        CognitiveProcessingMetrics,
        AnalyticsQuery,
        DataWarehouse,
        EventType,
        SessionStatus
    )

    from .etl import (
        ETLPipeline,
        DataTransformer,
        DataValidator,
        DataRetentionManager,
        ETLConfig
    )

    from .processor import (
        AnalyticsProcessor,
        RealTimeAnalytics,
        PredictiveAnalytics,
        AnomalyDetector
    )

    from .dashboard import (
        DashboardManager,
        VisualizationEngine,
        ReportGenerator
    )

    from .integrations import (
        BIIntegration,
        TableauConnector,
        PowerBIConnector,
        ExportManager,
        IntegrationManager,
        ExportConfig,
        BIConnection
    )

    from .service import (
        AnalyticsService,
        initialize_analytics_service,
        get_analytics_service,
        shutdown_analytics_service
    )

    try:
        from .flask_integration import (
            create_analytics_blueprint,
            integrate_analytics_with_app,
            add_analytics_middleware
        )
    except ImportError:
        # Flask integration optional
        pass

except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Analytics module import warning: {e}")
    pass

__version__ = "1.0.0"
__author__ = "Deep Tree Echo Analytics Team"

# Analytics configuration
ANALYTICS_CONFIG = {
    'retention_days': 365,
    'batch_size': 1000,
    'processing_interval': 60,  # seconds
    'anomaly_threshold': 2.0,
    'dashboard_refresh_rate': 30,  # seconds
    'export_formats': ['json', 'csv', 'xlsx', 'parquet']
}