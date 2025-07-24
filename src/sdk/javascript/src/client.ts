/**
 * Main HTTP client for the Deep Tree Echo API
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import {
  EchoClientConfig,
  CognitiveResult,
  SessionInfo,
  MemoryItem,
  SystemStatus,
  UsageAnalytics,
  QuotaInfo,
  APIResponse,
  ProcessingOptions,
  MemorySearchOptions,
  SessionConfiguration,
  BatchProcessingOptions
} from './types';
import {
  EchoAPIError,
  EchoAuthenticationError,
  EchoRateLimitError,
  EchoValidationError,
  EchoServerError,
  EchoNetworkError
} from './errors';

export class EchoClient {
  private httpClient: AxiosInstance;
  private quotaInfo: QuotaInfo | null = null;

  constructor(config: EchoClientConfig) {
    if (!config.apiKey) {
      throw new EchoAuthenticationError('API key is required');
    }

    this.httpClient = axios.create({
      baseURL: config.baseUrl || 'http://localhost:8000',
      timeout: config.timeout || 30000,
      headers: {
        'X-API-Key': config.apiKey,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': config.userAgent || '@deeptreeecho/sdk/1.0.0',
        'API-Version': config.apiVersion || 'v1'
      }
    });

    // Setup request interceptors
    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor for retry logic
    this.httpClient.interceptors.request.use(
      (config) => {
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling and quota tracking
    this.httpClient.interceptors.response.use(
      (response: AxiosResponse) => {
        // Update quota info from headers
        this.updateQuotaFromHeaders(response.headers);
        return response;
      },
      (error: AxiosError) => {
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private updateQuotaFromHeaders(headers: any): void {
    const tier = headers['x-ratelimit-tier'];
    const hourRemaining = headers['x-ratelimit-hour-remaining'];
    const dayRemaining = headers['x-ratelimit-day-remaining'];

    if (tier && hourRemaining && dayRemaining) {
      // This is a simplified implementation
      // In practice, you'd want to track more complete quota information
      this.quotaInfo = {
        tier,
        hourUsage: 0, // Would calculate from remaining
        hourLimit: 1000, // Would get from tier config
        dayUsage: 0, // Would calculate from remaining
        dayLimit: 10000, // Would get from tier config
        allowed: true
      };
    }
  }

  private handleError(error: AxiosError): Error {
    if (error.response) {
      const { status, data } = error.response;
      const message = (data as any)?.error || `HTTP ${status} error`;

      switch (status) {
        case 401:
          return new EchoAuthenticationError(message, status, data);
        case 429:
          const retryAfter = error.response.headers['retry-after'];
          return new EchoRateLimitError(message, retryAfter ? parseInt(retryAfter) : undefined, status, data);
        case 400:
          return new EchoValidationError(message, status, data);
        default:
          if (status >= 500) {
            return new EchoServerError(message, status, data);
          }
          return new EchoAPIError(message, status, data);
      }
    } else if (error.request) {
      return new EchoNetworkError('Network error: No response received', error);
    } else {
      return new EchoNetworkError(`Request error: ${error.message}`, error);
    }
  }

  private async makeRequest<T>(method: string, endpoint: string, data?: any, params?: any): Promise<T> {
    try {
      const response = await this.httpClient.request({
        method,
        url: `/api/v1/${endpoint.replace(/^\//, '')}`,
        data,
        params
      });

      const apiResponse: APIResponse<T> = response.data;
      
      if (!apiResponse.success) {
        throw new EchoAPIError(apiResponse.error || 'API request failed');
      }

      return apiResponse.data as T;
    } catch (error) {
      if (error instanceof EchoAPIError) {
        throw error;
      }
      throw this.handleError(error as AxiosError);
    }
  }

  // ========================================================================
  // System Operations
  // ========================================================================

  async getSystemStatus(): Promise<SystemStatus> {
    return this.makeRequest<SystemStatus>('GET', 'status');
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.httpClient.get('/health');
      return response.data?.status === 'healthy';
    } catch {
      return false;
    }
  }

  // ========================================================================
  // Cognitive Processing  
  // ========================================================================

  async processCognitiveInput(
    inputText: string,
    options: ProcessingOptions = {}
  ): Promise<CognitiveResult> {
    if (!inputText.trim()) {
      throw new EchoValidationError('Input text cannot be empty');
    }

    const requestData = {
      input: inputText,
      options: {
        temperature: options.temperature || 0.8,
        maxTokens: options.maxTokens || 2048,
        ...(options.sessionId && { sessionId: options.sessionId })
      }
    };

    return this.makeRequest<CognitiveResult>('POST', 'cognitive/process', requestData);
  }

  async batchProcess(
    inputs: string[],
    options: BatchProcessingOptions = {}
  ): Promise<CognitiveResult[]> {
    if (!inputs.length) {
      throw new EchoValidationError('Inputs array cannot be empty');
    }

    const concurrency = options.concurrency || 5;
    const delay = options.delayBetweenRequests || 100;
    const results: CognitiveResult[] = [];

    // Process in chunks to respect concurrency limits
    for (let i = 0; i < inputs.length; i += concurrency) {
      const chunk = inputs.slice(i, i + concurrency);
      
      const chunkPromises = chunk.map(async (input, index) => {
        if (index > 0 && delay > 0) {
          await new Promise(resolve => setTimeout(resolve, delay * index));
        }
        
        return this.processCognitiveInput(input, options);
      });

      const chunkResults = await Promise.all(chunkPromises);
      results.push(...chunkResults);

      // Small delay between chunks
      if (i + concurrency < inputs.length && delay > 0) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    return results;
  }

  // ========================================================================
  // Session Management
  // ========================================================================

  async createSession(
    configuration?: SessionConfiguration,
    metadata?: Record<string, any>
  ): Promise<SessionInfo> {
    const requestData: any = {};

    if (configuration) {
      Object.assign(requestData, {
        temperature: configuration.temperature,
        maxContextLength: configuration.maxContextLength,
        memoryPersistence: configuration.memoryPersistence
      });
    }

    if (metadata) {
      requestData.metadata = metadata;
    }

    return this.makeRequest<SessionInfo>('POST', 'sessions', requestData);
  }

  async getSession(sessionId: string): Promise<SessionInfo> {
    if (!sessionId) {
      throw new EchoValidationError('Session ID is required');
    }

    return this.makeRequest<SessionInfo>('GET', `sessions/${sessionId}`);
  }

  // ========================================================================
  // Memory Operations
  // ========================================================================

  async searchMemory(
    query: string,
    options: MemorySearchOptions = {}
  ): Promise<MemoryItem[]> {
    if (!query.trim()) {
      throw new EchoValidationError('Search query cannot be empty');
    }

    const requestData = {
      query,
      limit: options.limit || 10,
      minRelevance: options.minRelevance || 0.5,
      ...(options.memoryType && { memoryType: options.memoryType })
    };

    const response = await this.makeRequest<{results: MemoryItem[]}>('POST', 'memory/search', requestData);
    return response.results;
  }

  // ========================================================================
  // Analytics
  // ========================================================================

  async getUsageAnalytics(period: string = 'last_30_days'): Promise<UsageAnalytics> {
    return this.makeRequest<UsageAnalytics>('GET', 'analytics/usage', undefined, { period });
  }

  get quotaInfo(): QuotaInfo | null {
    return this.quotaInfo;
  }

  // ========================================================================
  // Utility Methods
  // ========================================================================

  async ping(): Promise<number> {
    const start = Date.now();
    await this.healthCheck();
    return Date.now() - start;
  }

  setApiVersion(version: string): void {
    this.httpClient.defaults.headers['API-Version'] = version;
  }

  setUserAgent(userAgent: string): void {
    this.httpClient.defaults.headers['User-Agent'] = userAgent;
  }
}