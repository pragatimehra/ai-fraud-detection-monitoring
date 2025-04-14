import { cn } from "../lib/utils";

type ScoreGaugeProps = {
  score: number;
  size?: "sm" | "md" | "lg";
  showValue?: boolean;
};

const ScoreGauge = ({
  score,
  size = "md",
  showValue = true,
}: ScoreGaugeProps) => {
  // Calculate the risk level and color based on score
  const getRiskLevel = (score: number) => {
    if (score < 0.4) return { level: "Low", colorClass: "fraud-risk-low" };
    if (score < 0.7) return { level: "Medium", colorClass: "fraud-risk-medium" };
    return { level: "High", colorClass: "fraud-risk-high" };
  };

  const { level, colorClass } = getRiskLevel(score);

  // Convert score to a percentage for the gauge
  const percentage = score * 100;
  
  // Calculate rotation for the gauge needle
  const rotation = (score * 180) - 90;
  
  // Size variations
  const sizeClasses = {
    sm: "w-24 h-12",
    md: "w-32 h-16",
    lg: "w-40 h-20",
  };
  
  const fontSizeClasses = {
    sm: "text-xs",
    md: "text-sm",
    lg: "text-base",
  };
  
  const needleSizeClasses = {
    sm: "h-10 w-1",
    md: "h-14 w-1.5",
    lg: "h-18 w-2",
  };

  return (
    <div className={cn("relative", sizeClasses[size])}>
      {/* Gauge background */}
      <div className="absolute w-full h-full rounded-t-full bg-muted overflow-hidden">
        <div className="absolute inset-0 flex items-end justify-center">
          <div className="w-full h-1/2 bg-gradient-to-r from-blue-400 via-orange-400 to-red-500 opacity-30"></div>
        </div>
      </div>
      
      {/* Gauge needle */}
      <div 
        className="absolute bottom-0 left-1/2 origin-bottom transition-transform duration-700"
        style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }}
      >
        <div className={cn("bg-white rounded", needleSizeClasses[size])}></div>
        <div className="w-3 h-3 rounded-full bg-white -mt-1.5 -ml-1"></div>
      </div>
      
      {/* Value display */}
      {showValue && (
        <div className="absolute bottom-0 left-0 w-full text-center pb-1">
          <div className={cn("font-bold", colorClass, fontSizeClasses[size])}>
            {level} Risk
          </div>
          <div className={cn("text-white opacity-90", fontSizeClasses[size])}>
            {percentage.toFixed(0)}%
          </div>
        </div>
      )}
    </div>
  );
};

export default ScoreGauge;
