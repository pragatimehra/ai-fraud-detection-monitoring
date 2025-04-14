
import { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { MessageSquare } from "lucide-react";
import { apiFeedback } from "@/utils/apiUtils";

interface FeedbackFormProps {
  onClose: () => void;
}

const FeedbackForm = ({ onClose }: FeedbackFormProps) => {
  const [feedback, setFeedback] = useState("");
  const [category, setCategory] = useState("false_positive");
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!feedback.trim()) {
      toast.error("Please provide feedback details");
      return;
    }
    
    setIsSubmitting(true);
    try {
      await apiFeedback.submitSystemFeedback(category, feedback);
      toast.success("Feedback submitted successfully");
      onClose();
    } catch (error) {
      console.error("Failed to submit feedback:", error);
      toast.error("Failed to submit feedback. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Provide Feedback
        </CardTitle>
        <CardDescription>
          Help us improve our fraud detection system
        </CardDescription>
      </CardHeader>
      
      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Feedback Category</label>
            <div className="flex flex-col space-y-1">
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  name="category"
                  value="false_positive"
                  checked={category === "false_positive"}
                  onChange={() => setCategory("false_positive")}
                  className="accent-primary"
                />
                <span className="text-sm">False Positive (incorrectly flagged as fraud)</span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  name="category"
                  value="false_negative"
                  checked={category === "false_negative"}
                  onChange={() => setCategory("false_negative")}
                  className="accent-primary"
                />
                <span className="text-sm">False Negative (missed actual fraud)</span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  name="category"
                  value="model_feedback"
                  checked={category === "model_feedback"}
                  onChange={() => setCategory("model_feedback")}
                  className="accent-primary"
                />
                <span className="text-sm">Model Performance Feedback</span>
              </label>
            </div>
          </div>
          
          <Separator />
          
          <div className="space-y-2">
            <label htmlFor="feedback" className="text-sm font-medium">
              Feedback Details
            </label>
            <textarea
              id="feedback"
              className="w-full h-32 p-3 bg-secondary text-sm rounded border border-border resize-none focus:outline-none focus:ring-1 focus:ring-primary"
              placeholder="Provide details about why this prediction is incorrect or how we can improve..."
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              required
            />
          </div>
        </CardContent>
        
        <CardFooter className="flex justify-end gap-3">
          <Button type="button" variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Submitting..." : "Submit Feedback"}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
};

export default FeedbackForm;
