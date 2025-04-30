
// API utilities for connecting to FastAPI backend

const API_URL = "http://localhost:8001/api";

// Mock data for when the backend isn't available
const MOCK_TRANSACTIONS = [
  {
    "id": "T123456",
    "timestamp": "2025-04-07T09:15:30",
    "amount": 15000,
    "accountNumber": "8765432109",
    "transactionType": "wire",
    "score": 0.85,
    "reason": "Transaction T123456 was flagged because the amount of ₹15,000 exceeds typical daily averages by over 300% for this account. The transaction also occurred from an unusual IP address not previously associated with this account."
  },
  {
    "id": "T123457",
    "timestamp": "2025-04-07T10:23:45",
    "amount": 5999,
    "accountNumber": "7654321098",
    "transactionType": "card",
    "score": 0.62,
    "reason": "Transaction T123457 was flagged due to unusual merchant category. This account typically performs transactions in retail and grocery, but this payment was made to a high-risk merchant category in a foreign jurisdiction."
  },
  {
    "id": "T123458",
    "timestamp": "2025-04-07T14:56:12",
    "amount": 2500,
    "accountNumber": "6543210987",
    "transactionType": "online",
    "score": 0.35,
    "reason": "Transaction T123458 triggered a low-level alert due to being slightly outside normal spending patterns. The transaction amount is higher than average for this merchant type, but remains within reasonable bounds for the account history."
  },
  {
    "id": "T123459",
    "timestamp": "2025-04-06T18:34:22",
    "amount": 12500,
    "accountNumber": "5432109876",
    "transactionType": "atm",
    "score": 0.78,
    "reason": "Transaction T123459 was flagged as potentially fraudulent because this ATM withdrawal occurred in a location over 500km from the account holder's typical transaction area. Additionally, there were 3 failed PIN attempts before this successful transaction."
  },
  {
    "id": "T123460",
    "timestamp": "2025-04-06T11:45:33",
    "amount": 8750,
    "accountNumber": "4321098765",
    "transactionType": "pos",
    "score": 0.52,
    "reason": "Transaction T123460 raised moderate concern due to unusual timing. This transaction occurred at 11:45 PM, while the account holder typically makes purchases between 8 AM and 8 PM. The merchant category itself is not unusual for this customer."
  }
];

// Generic fetch wrapper with error handling
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_URL}${endpoint}`;
  
  console.log(`Making API request to: ${url}`);
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      // Important for CORS preflight to work correctly
      credentials: 'omit', 
      mode: 'cors',
    });

    console.log(`API response status: ${response.status}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API error: ${response.status}`, errorText);
      throw new Error(errorText || `API error: ${response.status}`);
    }

    const data = await response.json();
    console.log("API response data:", data);
    return data;
  } catch (error) {
    console.error("API request failed:", error);
    throw error;
  }
}

// Authentication
export const apiAuth = {
  login: async (username: string, password: string) => {
    return fetchAPI<{ success: boolean; message?: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
  },
};

// Transactions
export const apiTransactions = {
  getAll: async () => {
    try {
      // Try to fetch from API first
      return await fetchAPI<any[]>("/transactions");
    } catch (error) {
      console.warn("Failed to fetch from API, using mock data:", error);
      // Simulate a brief delay to mimic network latency
      await new Promise(resolve => setTimeout(resolve, 500));
      return [...MOCK_TRANSACTIONS];
    }
  },
};

// Feedback
export const apiFeedback = {
  submitTransactionFeedback: async (
    transactionId: string,
    isCorrect: boolean,
    feedback?: string
  ) => {
    try {
      return await fetchAPI("/feedback/transaction", {
        method: "POST",
        body: JSON.stringify({
          transaction_id: transactionId,
          is_correct: isCorrect,
          feedback,
        }),
      });
    } catch (error) {
      console.warn("Failed to submit feedback to API, simulating success:", error);
      // Return a mock successful response
      return { success: true, message: "Feedback submitted (offline mode)" };
    }
  },
  submitSystemFeedback: async (category: string, details: string) => {
    try {
      return await fetchAPI("/feedback/system", {
        method: "POST",
        body: JSON.stringify({
          category,
          details,
        }),
      });
    } catch (error) {
      console.warn("Failed to submit system feedback to API, simulating success:", error);
      // Return a mock successful response
      return { success: true, message: "System feedback submitted (offline mode)" };
    }
  },
};
