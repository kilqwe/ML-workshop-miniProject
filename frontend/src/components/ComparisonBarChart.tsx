"use client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

interface BarChartProps {
  data: { name: string; rating: number }[];
}

// A set of distinct colors for the bars
const COLORS = ["#3b82f6", "#22c55e", "#f97316", "#ef4444", "#8b5cf6"];

export function ComparisonBarChart({ data }: BarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[60, 100]} />
        <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
        <Tooltip />
        <Bar dataKey="rating" name="Overall Rating"> {/* The 'name' prop is used by the Tooltip */}
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}