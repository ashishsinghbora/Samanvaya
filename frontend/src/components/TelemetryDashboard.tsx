import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart } from 'recharts';
import { ShieldAlert, ShieldCheck, Activity, RefreshCw } from 'lucide-react';

export const TelemetryDashboard: React.FC = () => {
  const [data, setData] = useState([
    { name: 'Align 1', rmse: 0.35, anomaly: false, conf: 98 },
    { name: 'Align 2', rmse: 0.28, anomaly: false, conf: 99 },
    { name: 'Align 3', rmse: 2.50, anomaly: true, conf: 45 },
    { name: 'Align 4', rmse: 0.41, anomaly: false, conf: 95 },
  ]);

  const [isRefreshing, setIsRefreshing] = useState(false);

  const simulateNewTelemetry = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setData(prev => {
        const newData = [...prev.slice(1)]; // remove first item
        const isAnomaly = Math.random() > 0.8;
        newData.push({
          name: `Align ${parseInt(prev[prev.length-1].name.split(' ')[1]) + 1}`,
          rmse: isAnomaly ? +(Math.random() * 3 + 1.5).toFixed(2) : +(Math.random() * 0.5 + 0.1).toFixed(2),
          anomaly: isAnomaly,
          conf: isAnomaly ? Math.floor(Math.random() * 40 + 30) : Math.floor(Math.random() * 10 + 90)
        });
        return newData;
      });
      setIsRefreshing(false);
    }, 600);
  };

  // Auto refresh every 5 seconds for demo purposes
  useEffect(() => {
    const interval = setInterval(simulateNewTelemetry, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-6 bg-lunar-card backdrop-blur-xl rounded-2xl border border-slate-700/50 shadow-2xl text-white mb-8"
    >
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-sky-500/20 rounded-lg border border-sky-500/30">
            <Activity className="text-sky-400" size={24} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white tracking-wide">IsolationForest Anomalies</h2>
            <p className="text-xs text-slate-400">Live RMSE error tracking across photogrammetry nodes</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <button 
            onClick={simulateNewTelemetry}
            className="flex items-center gap-2 text-sm bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg border border-slate-600 transition-colors"
          >
            <RefreshCw size={14} className={isRefreshing ? "animate-spin" : ""} />
            Poll Server
          </button>
          <div className="flex items-center gap-2 text-sm text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-3 py-2 rounded-lg shadow-[0_0_10px_rgba(52,211,153,0.1)]">
            <ShieldCheck size={16} /> DevSecOps Secured
          </div>
        </div>
      </div>
      
      <div className="h-64 w-full mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorRmse" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#38bdf8" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="name" stroke="#64748b" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
            <YAxis stroke="#64748b" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid #334155', borderRadius: '8px', color: '#fff' }} 
              itemStyle={{ color: '#38bdf8' }}
            />
            <Area type="monotone" dataKey="rmse" stroke="#38bdf8" strokeWidth={3} fillOpacity={1} fill="url(#colorRmse)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {data.map((item) => (
          <motion.div 
            key={item.name}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            whileHover={{ scale: 1.02, y: -2 }}
            className={`p-4 rounded-xl flex flex-col justify-between ${
              item.anomaly 
                ? 'bg-rose-500/10 border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.15)]' 
                : 'bg-slate-800/40 border-slate-700/50 hover:bg-slate-800/60'
            } border transition-all`}
          >
            <div className="flex justify-between items-start mb-2">
              <span className="text-sm font-semibold text-slate-300">{item.name}</span>
              {item.anomaly ? <ShieldAlert size={16} className="text-rose-400" /> : <ShieldCheck size={16} className="text-emerald-400" />}
            </div>
            
            <div>
              <div className="text-2xl font-bold flex items-baseline gap-1">
                <span className={item.anomaly ? 'text-rose-400' : 'text-sky-400'}>{item.rmse}</span>
                <span className="text-xs text-slate-500 font-normal">RMSE</span>
              </div>
              <div className="text-xs mt-1 text-slate-400 flex items-center justify-between">
                <span>Confidence:</span>
                <span className={item.conf < 80 ? 'text-amber-400' : 'text-emerald-400'}>{item.conf}%</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};
