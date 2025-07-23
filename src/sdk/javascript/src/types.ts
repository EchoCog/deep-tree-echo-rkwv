/**
 * Type definitions for the Deep Tree Echo API
 */

export interface EchoClientConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  userAgent?: string;
  apiVersion?: string;
}

export interface MembraneOutput {
  membraneType: string;
  response: string;
  confidence: number;
  processingTime: number;
}

export interface CognitiveState {
  declarativeMemoryItems: number;
  proceduralMemoryItems: number;
  episodicMemoryItems: number;
  temporalContextLength: number;
  currentGoals: number;
  lastUpdated?: string;
}

export interface CognitiveResult {
  inputText: string;
  integratedResponse: string;
  processingTime: number;
  sessionId: string;
  membraneOutputs?: MembraneOutput[];
  cognitiveState?: CognitiveState;
  timestamp?: string;
  confidence?: number;
}

export interface SessionConfiguration {
  temperature?: number;
  maxContextLength?: number;
  memoryPersistence?: boolean;
}

export interface SessionInfo {
  sessionId: string;
  status: string;
  createdAt: string;
  lastActivity: string;
  messageCount: number;
  totalTokensProcessed: number;
  configuration?: SessionConfiguration;
  cognitiveState?: CognitiveState;
  metadata?: Record<string, any>;
}

export interface MemoryItem {
  id: string;
  content: string;
  memoryType: string;
  relevanceScore: number;
  createdAt: string;
  lastAccessed: string;
  accessCount: number;
  metadata?: Record<string, any>;
}

export interface SystemServices {
  cognitiveProcessing: boolean;
  memorySystem: boolean;
  apiServer: boolean;
}

export interface SystemPerformance {
  responseTimeMs: number;
  throughputRpm: number;
  cacheHitRate: number;
}

export interface SystemInfo {
  status: string;
  version: string;
  uptime: number;
  echoSystemInitialized: boolean;
}

export interface SystemStatus {
  system: SystemInfo;
  services: SystemServices;
  performance: SystemPerformance;
}

export interface UsageAnalytics {
  totalRequests: number;
  successfulRequests: number;
  errorRequests: number;
  averageResponseTime: number;
  apiTier: string;
  quotaRemaining: number;
  period: string;
}

export interface QuotaInfo {
  tier: string;
  hourUsage: number;
  hourLimit: number;
  dayUsage: number;
  dayLimit: number;
  allowed: boolean;
}

export interface APIResponse<T = any> {
  success: boolean;
  timestamp: string;
  version: string;
  data?: T;
  error?: string;
  meta?: Record<string, any>;
}

export interface ProcessingOptions {
  temperature?: number;
  maxTokens?: number;
  sessionId?: string;
}

export interface MemorySearchOptions {
  memoryType?: string;
  limit?: number;
  minRelevance?: number;
}

export interface BatchProcessingOptions extends ProcessingOptions {
  concurrency?: number;
  delayBetweenRequests?: number;
}

// GraphQL Types
export interface GraphQLQuery {
  query: string;
  variables?: Record<string, any>;
  operationName?: string;
}

export interface GraphQLResponse<T = any> {
  data?: T;
  errors?: Array<{
    message: string;
    locations?: Array<{
      line: number;
      column: number;
    }>;
    path?: string[];
  }>;
}

// WebSocket Types
export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface WebSocketConfig {
  url: string;
  apiKey: string;
  protocols?: string[];
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

// Event Types
export type EchoEventType = 
  | 'connected'
  | 'disconnected'
  | 'message'
  | 'error'
  | 'cognitive_result'
  | 'memory_update'
  | 'session_update';

export interface EchoEvent {
  type: EchoEventType;
  data: any;
  timestamp: string;
}