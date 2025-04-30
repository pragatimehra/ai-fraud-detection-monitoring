import { useState, useEffect } from "react";

import Navbar from "../components/Navbar";

import TransactionCard, { Transaction } from "../components/TransactionCard";

import ScoreGauge from "../components/ScoreGauge";

import FeedbackForm from "../components/FeedbackForm";

import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

import { Button } from "../components/ui/button";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";

import { Separator } from "../components/ui/separator";

import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";

import { toast } from "sonner";

import { apiTransactions, apiFeedback } from "../utils/apiUtils";

import {

  FileDown,

  Filter,

  AlertTriangle,

  PieChart,

  BarChart3,

  Clock,

  Search,

  RefreshCw

} from "lucide-react";
 
const Dashboard = () => {

  const [transactions, setTransactions] = useState<Transaction[]>([]);

  const [isLoading, setIsLoading] = useState(true);

  const [feedbackVisible, setFeedbackVisible] = useState(false);

  const [riskFilter, setRiskFilter] = useState<"all" | "high" | "medium" | "low">("all");

  const [dateFilter, setDateFilter] = useState<string>(""); // format: YYYY-MM-DD

  const [anomalyFilter, setAnomalyFilter] = useState<string>(""); // free text filter for anomaly type

  const [refreshing, setRefreshing] = useState(false);
 
  // Fetch transactions

  useEffect(() => {

    const fetchTransactions = async () => {

      try {

        setIsLoading(true);

        const data = await apiTransactions.getAll();

        setTransactions(data);

      } catch (error) {

        console.error("Failed to fetch transactions:", error);

        toast.error("Failed to load transactions. Please try again.");

      } finally {

        setIsLoading(false);

      }

    };
 
    fetchTransactions();

  }, []);
 
  // Combine filters: risk level, date, and anomaly type

  const filteredTransactions = transactions.filter(transaction => {

    // Filter by risk level

    let passesRisk = false;

    if (riskFilter === "all") {

      passesRisk = true;

    } else if (riskFilter === "high") {

      passesRisk = transaction.score >= 0.7;

    } else if (riskFilter === "medium") {

      passesRisk = transaction.score >= 0.4 && transaction.score < 0.7;

    } else if (riskFilter === "low") {

      passesRisk = transaction.score < 0.4;

    }
 
    // Filter by date (if a date is selected)

    let passesDate = true;

    if (dateFilter) {

      const transactionDate = new Date(transaction.timestamp).toISOString().split("T")[0];

      passesDate = transactionDate === dateFilter;

    }
 
    // Filter by anomaly type (if provided)

    let passesAnomaly = true;

    if (anomalyFilter.trim() !== "") {

      passesAnomaly = transaction.reason.toLowerCase().includes(anomalyFilter.toLowerCase());

    }
 
    return passesRisk && passesDate && passesAnomaly;

  });
 
  // Count transactions by risk level (for the stats cards)

  const highRiskCount = transactions.filter(t => t.score >= 0.7).length;

  const mediumRiskCount = transactions.filter(t => t.score >= 0.4 && t.score < 0.7).length;

  const lowRiskCount = transactions.filter(t => t.score < 0.4).length;
 
  // Handle feedback submission for each transaction

  const handleFeedbackSubmit = async (id: string, isCorrect: boolean, feedback?: string) => {

    try {

      await apiFeedback.submitTransactionFeedback(id, isCorrect, feedback);

      toast.success("Feedback submitted successfully");

      // Mark feedback as given for the transaction locally.

      setTransactions(prev =>

        prev.map(t => (t.id === id ? { ...t, feedbackGiven: true } : t))

      );

    } catch (error) {

      console.error("Failed to submit feedback:", error);

      toast.error("Failed to submit feedback. Please try again.");

    }

  };
 
  // Handle system feedback submission

  const handleSystemFeedback = async (category: string, details: string) => {

    try {

      await apiFeedback.submitSystemFeedback(category, details);

      setFeedbackVisible(false);

      toast.success("System feedback submitted successfully");

    } catch (error) {

      console.error("Failed to submit system feedback:", error);

      toast.error("Failed to submit feedback. Please try again.");

    }

  };
 
  // Handle CSV download (only include transactions with feedback given)

  const handleDownloadCSV = () => {

    const headers = [

      "Transaction ID",

      "Timestamp",

      "Amount",

      "Account",

      "Type",

      "Risk Score",

      "AI Analysis"

    ];

    const csvRows = [

      headers.join(","),

      ...transactions

        .filter(t => t.feedbackGiven)

        .map(t => [

          t.id,

          t.timestamp,

          t.amount,

          t.accountNumber,

          t.transactionType,

          t.score,

          `"${t.reason.replace(/"/g, '""')}"`

        ].join(","))

    ];

    const csvContent = csvRows.join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.setAttribute("href", url);

    link.setAttribute("download", `fraud_log_${new Date().toISOString().split("T")[0]}.csv`);

    link.style.visibility = "hidden";

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    toast.success("CSV file downloaded successfully");

  };
 
  // Refresh data from API

  const handleRefresh = async () => {

    setRefreshing(true);

    try {

      const data = await apiTransactions.getAll();

      setTransactions(data);

      toast.success("Dashboard refreshed with latest data");

    } catch (error) {

      console.error("Failed to refresh data:", error);

      toast.error("Failed to refresh data. Please try again.");

    } finally {

      setRefreshing(false);

    }

  };
 
  return (
<div className="min-h-screen flex flex-col">
<Navbar />
<main className="flex-1 container py-6 px-4">
<div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
<div>
<h1 className="text-2xl font-bold">Fraud Detection Dashboard</h1>
<p className="text-muted-foreground">Monitor and manage suspicious activity</p>
</div>
<div className="flex gap-2">
<Button

              variant="outline"

              className="gap-1"

              onClick={handleDownloadCSV}
>
<FileDown className="h-4 w-4" />

              Export CSV
</Button>
<Button

              className="gap-1"

              onClick={handleRefresh}

              disabled={refreshing}
>
<RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />

              Refresh
</Button>
</div>
</div>
 
        {/* Stats Cards */}
<div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
<Card>
<CardHeader className="py-4">
<CardTitle className="text-base flex justify-between items-center">
<span className="flex items-center gap-2">
<AlertTriangle className="h-5 w-5 text-fraud-high" />

                  High Risk
</span>
<span className="text-fraud-high font-bold">{highRiskCount}</span>
</CardTitle>
</CardHeader>
</Card>
<Card>
<CardHeader className="py-4">
<CardTitle className="text-base flex justify-between items-center">
<span className="flex items-center gap-2">
<AlertTriangle className="h-5 w-5 text-fraud-medium" />

                  Medium Risk
</span>
<span className="text-fraud-medium font-bold">{mediumRiskCount}</span>
</CardTitle>
</CardHeader>
</Card>
<Card>
<CardHeader className="py-4">
<CardTitle className="text-base flex justify-between items-center">
<span className="flex items-center gap-2">
<AlertTriangle className="h-5 w-5 text-fraud-low" />

                  Low Risk
</span>
<span className="text-fraud-low font-bold">{lowRiskCount}</span>
</CardTitle>
</CardHeader>
</Card>
</div>
 
        {/* Main Content */}

        {isLoading ? (
<div className="flex justify-center items-center h-64">
<p className="text-lg">Loading transactions...</p>
</div>

        ) : (
<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
<div className="lg:col-span-2">
<div className="bg-card rounded-lg border border-border mb-6">
<div className="p-4 flex flex-col md:flex-row justify-between md:items-center gap-2">
<h2 className="text-xl font-semibold">Flagged Transactions</h2>
<div className="flex gap-2 items-center">
<div className="relative flex items-center">
<Search className="absolute left-2.5 h-4 w-4 text-muted-foreground" />
<input

                        type="text"

                        placeholder="Search transactions..."

                        className="pl-8 h-9 w-full md:w-[200px] bg-secondary rounded-md text-sm border border-border focus:outline-none focus:ring-1 focus:ring-primary"

 
                      />
</div>
<Button variant="outline" size="icon">
<Filter className="h-4 w-4" />
</Button>
</div>
</div>

                {/* New filter controls for date and anomaly type */}
<div className="p-4 flex flex-col md:flex-row gap-4 items-center border-b border-border">
<div className="flex flex-col">
<label className="text-xs text-muted-foreground">Filter by Date</label>
<input

                      type="date"

                      value={dateFilter}

                      onChange={(e) => setDateFilter(e.target.value)}

                      className="h-9 w-full md:w-[200px] bg-secondary rounded-md text-sm border border-border focus:outline-none focus:ring-1 focus:ring-primary"

                    />
</div>
<div className="flex flex-col">
<label className="text-xs text-muted-foreground">Filter by Anomaly Type</label>
<input

                      type="text"

                      placeholder="e.g. duplicate, unusual"

                      value={anomalyFilter}

                      onChange={(e) => setAnomalyFilter(e.target.value)}

                      className="h-9 w-full md:w-[200px] bg-secondary rounded-md text-sm border border-border focus:outline-none focus:ring-1 focus:ring-primary"

                    />
</div>
</div>
<Tabs defaultValue="all" className="w-full">
<div className="px-4 border-b border-border">
<TabsList className="bg-transparent">
<TabsTrigger value="all" onClick={() => setRiskFilter("all")}>

                        All
</TabsTrigger>
<TabsTrigger value="high" onClick={() => setRiskFilter("high")}>

                        High Risk
</TabsTrigger>
<TabsTrigger value="medium" onClick={() => setRiskFilter("medium")}>

                        Medium Risk
</TabsTrigger>
<TabsTrigger value="low" onClick={() => setRiskFilter("low")}>

                        Low Risk
</TabsTrigger>
</TabsList>
</div>

                  {/* Each TabsContent now renders the (dynamically filtered) transactions */}
<TabsContent value="all" className="p-0 m-0">
<div className="p-4 space-y-4">

                      {filteredTransactions.length > 0 ? (

                        filteredTransactions.map(transaction => (
<TransactionCard

                            key={transaction.id}

                            transaction={transaction}

                            onFeedbackSubmit={handleFeedbackSubmit}

                          />

                        ))

                      ) : (
<Alert>
<AlertTitle>No transactions found</AlertTitle>
<AlertDescription>

                            No transactions match your current filter criteria.
</AlertDescription>
</Alert>

                      )}
</div>
</TabsContent>
<TabsContent value="high" className="p-0 m-0">
<div className="p-4 space-y-4">

                      {filteredTransactions.length > 0 ? (

                        filteredTransactions.map(transaction => (
<TransactionCard

                            key={transaction.id}

                            transaction={transaction}

                            onFeedbackSubmit={handleFeedbackSubmit}

                          />

                        ))

                      ) : (
<Alert>
<AlertTitle>No transactions found</AlertTitle>
<AlertDescription>

                            No transactions match your current filter criteria.
</AlertDescription>
</Alert>

                      )}
</div>
</TabsContent>
<TabsContent value="medium" className="p-0 m-0">
<div className="p-4 space-y-4">

                      {filteredTransactions.length > 0 ? (

                        filteredTransactions.map(transaction => (
<TransactionCard

                            key={transaction.id}

                            transaction={transaction}

                            onFeedbackSubmit={handleFeedbackSubmit}

                          />

                        ))

                      ) : (
<Alert>
<AlertTitle>No transactions found</AlertTitle>
<AlertDescription>

                            No transactions match your current filter criteria.
</AlertDescription>
</Alert>

                      )}
</div>
</TabsContent>
<TabsContent value="low" className="p-0 m-0">
<div className="p-4 space-y-4">

                      {filteredTransactions.length > 0 ? (

                        filteredTransactions.map(transaction => (
<TransactionCard

                            key={transaction.id}

                            transaction={transaction}

                            onFeedbackSubmit={handleFeedbackSubmit}

                          />

                        ))

                      ) : (
<Alert>
<AlertTitle>No transactions found</AlertTitle>
<AlertDescription>

                            No transactions match your current filter criteria.
</AlertDescription>
</Alert>

                      )}
</div>
</TabsContent>
</Tabs>
</div>
</div>
<div className="space-y-6">

              {/* Risk Distribution Card */}
<Card>
<CardHeader className="pb-2">
<CardTitle className="text-base flex items-center gap-2">
<PieChart className="h-4 w-4" />

                    Fraud Risk Distribution
</CardTitle>
</CardHeader>
<CardContent>
<div className="flex justify-center py-4">
<ScoreGauge score={0.68} size="lg" showValue={false} />
</div>
<div className="grid grid-cols-3 gap-2 text-center pt-2">
<div>
<div className="h-2 rounded-full bg-fraud-low mb-1"></div>
<p className="text-xs text-muted-foreground">Low Risk</p>
<p className="text-sm font-medium">{lowRiskCount}</p>
</div>
<div>

 
                      <div className="h-2 rounded-full bg-fraud-medium mb-1"></div>
<p className="text-xs text-muted-foreground">Medium Risk</p>
<p className="text-sm font-medium">{mediumRiskCount}</p>
</div>
<div>
<div className="h-2 rounded-full bg-fraud-high mb-1"></div>
<p className="text-xs text-muted-foreground">High Risk</p>
<p className="text-sm font-medium">{highRiskCount}</p>
</div>
</div>
</CardContent>
</Card>

              {/* Recent Activity Card */}
<Card>
<CardHeader className="pb-2">
<CardTitle className="text-base flex items-center gap-2">
<Clock className="h-4 w-4" />

                    Recent Activity
</CardTitle>
</CardHeader>
<CardContent className="px-2">
<div className="space-y-1">
<div className="text-xs p-2 hover:bg-secondary rounded flex items-center gap-2">
<div className="w-2 h-2 rounded-full bg-fraud-high"></div>
<p className="flex-1">High risk transaction detected</p>
<p className="text-muted-foreground">4min ago</p>
</div>
<div className="text-xs p-2 hover:bg-secondary rounded flex items-center gap-2">
<div className="w-2 h-2 rounded-full bg-primary"></div>
<p className="flex-1">System updated transaction risk models</p>
<p className="text-muted-foreground">15min ago</p>
</div>
<div className="text-xs p-2 hover:bg-secondary rounded flex items-center gap-2">
<div className="w-2 h-2 rounded-full bg-fraud-medium"></div>
<p className="flex-1">Medium risk transaction flagged</p>
<p className="text-muted-foreground">32min ago</p>
</div>
<div className="text-xs p-2 hover:bg-secondary rounded flex items-center gap-2">
<div className="w-2 h-2 rounded-full bg-primary"></div>
<p className="flex-1">Admin logged in</p>
<p className="text-muted-foreground">45min ago</p>
</div>
</div>
</CardContent>
</Card>

              {/* Transaction Volume Card */}
<Card>
<CardHeader className="pb-2">
<CardTitle className="text-base flex items-center gap-2">
<BarChart3 className="h-4 w-4" />

                    Transaction Volume
</CardTitle>
</CardHeader>
<CardContent>
<div className="h-32 flex items-end justify-between gap-1 pt-2">

                    {[15, 32, 26, 42, 38, 55, 29].map((value, i) => (
<div key={i} className="w-full">
<div

                          className="bg-primary/70 hover:bg-primary rounded-t"

                          style={{ height: `${value * 1.5}px` }}
></div>
</div>

                    ))}
</div>
<div className="flex justify-between text-xs text-muted-foreground pt-1">
<div>Mon</div>
<div>Tue</div>
<div>Wed</div>
<div>Thu</div>
<div>Fri</div>
<div>Sat</div>
<div>Sun</div>
</div>
</CardContent>
</Card>

              {/* Feedback Button */}
<Button

                variant="outline"

                className="w-full"

                onClick={() => setFeedbackVisible(true)}
>

                Provide System Feedback
</Button>

              {/* Feedback Form Modal */}

              {feedbackVisible && (
<div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
<div className="max-w-lg w-full">
<FeedbackForm onClose={() => setFeedbackVisible(false)} />
</div>
</div>

              )}
</div>
</div>

        )}
</main>
</div>

  );

};
 
export default Dashboard;

 