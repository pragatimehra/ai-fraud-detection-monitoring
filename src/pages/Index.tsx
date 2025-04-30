
import { Navigate } from "react-router-dom";

const Index = () => {
  // Just redirect directly to dashboard
  return <Navigate to="/dashboard" replace />;
};

export default Index;
