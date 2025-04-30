
// Modified auth utils that skips authentication

// Always return true for authentication check
export const isAuthenticated = (): boolean => {
  return true;
};

// No-op login function
export const login = async (username: string, password: string): Promise<boolean> => {
  console.log("Login bypassed, automatically authenticated");
  return true;
};

// No-op logout function
export const logout = (): void => {
  console.log("Logout bypassed, remaining authenticated");
  // Don't redirect since we're always authenticated
};
