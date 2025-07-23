/**
 * Error classes for the Deep Tree Echo SDK
 */

export class EchoAPIError extends Error {
  public statusCode?: number;
  public response?: any;

  constructor(message: string, statusCode?: number, response?: any) {
    super(message);
    this.name = 'EchoAPIError';
    this.statusCode = statusCode;
    this.response = response;
  }
}

export class EchoAuthenticationError extends EchoAPIError {
  constructor(message: string, statusCode?: number, response?: any) {
    super(message, statusCode, response);
    this.name = 'EchoAuthenticationError';
  }
}

export class EchoRateLimitError extends EchoAPIError {
  public retryAfter?: number;

  constructor(message: string, retryAfter?: number, statusCode?: number, response?: any) {
    super(message, statusCode, response);
    this.name = 'EchoRateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class EchoQuotaExceededError extends EchoRateLimitError {
  constructor(message: string, retryAfter?: number, statusCode?: number, response?: any) {
    super(message, retryAfter, statusCode, response);
    this.name = 'EchoQuotaExceededError';
  }
}

export class EchoValidationError extends EchoAPIError {
  constructor(message: string, statusCode?: number, response?: any) {
    super(message, statusCode, response);
    this.name = 'EchoValidationError';
  }
}

export class EchoServerError extends EchoAPIError {
  constructor(message: string, statusCode?: number, response?: any) {
    super(message, statusCode, response);
    this.name = 'EchoServerError';
  }
}

export class EchoNetworkError extends EchoAPIError {
  constructor(message: string, originalError?: Error) {
    super(message);
    this.name = 'EchoNetworkError';
    if (originalError) {
      this.stack = originalError.stack;
    }
  }
}

export class EchoWebSocketError extends EchoAPIError {
  constructor(message: string, code?: number) {
    super(message, code);
    this.name = 'EchoWebSocketError';
  }
}