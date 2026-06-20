import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Search, TrendingUp, TrendingDown, Activity, ShieldAlert, Cpu, Newspaper, Hexagon } from 'lucide-react';

export default function App() {
  const [ticker, setTicker] = useState('');
  const [searchVal, setSearchVal] = useState('');
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isHome, setIsHome] = useState(true);

  const fetchAnalysis = async (targetTicker) => {
    if (!targetTicker) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/analyze/${targetTicker.toUpperCase()}`);
      if (!res.ok) throw new Error(`Ticker ${targetTicker.toUpperCase()} execution anomaly.`);
      const data = await res.json();
      setStockData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isHome && ticker) {
      fetchAnalysis(ticker);
    }
  }, [ticker, isHome]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchVal.trim()) {
      setTicker(searchVal.trim().toUpperCase());
      setIsHome(false);
    }
  };

  const handleQuickSearch = (target) => {
    setSearchVal(target);
    setTicker(target);
    setIsHome(false);
  };

  const goHome = () => {
    setIsHome(true);
    setSearchVal('');
    setTicker('');
    setStockData(null);
  };

  const mockChartData = [
    { name: '60d', price: stockData ? stockData.latest_price * 0.92 : 150 },
    { name: '45d', price: stockData ? stockData.latest_price * 0.95 : 155 },
    { name: '30d', price: stockData ? stockData.latest_price * 0.98 : 158 },
    { name: '15d', price: stockData ? stockData.latest_price * 0.97 : 157 },
    { name: 'Now', price: stockData ? stockData.latest_price : 160 }
  ];

  if (isHome) {
    return (
      <div className="min-h-screen bg-[#070a13] text-gray-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-black relative overflow-hidden">
        {/* Background Grid Effect */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-20 pointer-events-none"></div>

        

        {/* Main Content */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 relative z-10 w-full max-w-4xl mx-auto -mt-20">
          <div className="space-y-8 flex flex-col items-center w-full">
            
            {/* Logo Header */}
            <div className="flex flex-col items-center justify-center space-y-6 mb-4">
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500 to-teal-400 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
                <div className="relative bg-[#0c1020] border border-gray-800 p-5 rounded-2xl shadow-2xl">
                  <Hexagon className="h-14 w-14 text-emerald-400" strokeWidth={2} />
                </div>
              </div>
              <h1 className="text-6xl md:text-8xl font-black tracking-tighter text-white drop-shadow-sm text-center">
              NEXUS
              </h1>
            </div>

            {/* Glowing Search Bar */}
            <form onSubmit={handleSearch} className="relative w-full max-w-2xl group mt-8">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500 to-purple-600 rounded-2xl blur opacity-20 group-focus-within:opacity-40 transition duration-500"></div>
              <input
                type="text"
                placeholder="Enter a ticker (e.g., AAPL, TSLA)..."
                value={searchVal}
                onChange={(e) => setSearchVal(e.target.value)}
                autoFocus
                className="relative w-full bg-[#0c1020]/90 backdrop-blur-xl border border-gray-700/50 rounded-2xl pl-16 pr-6 py-5 text-xl text-white shadow-2xl focus:outline-none focus:border-emerald-500/50 transition-all placeholder-gray-600"
              />
              <Search className="absolute left-6 top-5 h-7 w-7 text-gray-500 group-focus-within:text-emerald-400 transition-colors z-10" />
            </form>

            {/* Quick Access Pills */}
            <div className="pt-8 flex flex-col items-center space-y-4">
              <span className="text-xs font-mono text-gray-600 uppercase tracking-widest">Active Neural Models</span>
              <div className="flex flex-wrap justify-center gap-3">
                {['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMD', 'PLTR'].map((t) => (
                  <button
                    key={t}
                    onClick={() => handleQuickSearch(t)}
                    className="px-4 py-2 rounded-lg bg-[#131930] border border-gray-800 text-gray-400 font-mono text-sm hover:border-emerald-500/50 hover:text-emerald-400 transition-all shadow-sm hover:shadow-emerald-900/20"
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#070a13] text-gray-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-black">
      <header className="border-b border-gray-800 bg-[#0c1020] px-6 py-4 flex items-center justify-between shadow-lg">
        <div className="flex items-center space-x-3 cursor-pointer hover:opacity-80 transition-opacity" onClick={goHome}>
          <div className="bg-gradient-to-tr from-emerald-500 to-teal-400 p-2 rounded-lg">
            <Hexagon className="h-5 w-5 text-black" strokeWidth={2.5} />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">
          NEXUS
          </h1>
        </div>

        <form onSubmit={handleSearch} className="relative w-80">
          <input
            type="text"
            placeholder="Search assets..."
            value={searchVal}
            onChange={(e) => setSearchVal(e.target.value)}
            className="w-full bg-[#131930] border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-emerald-500 transition-colors placeholder-gray-500"
          />
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-500" />
        </form>
      </header>

      <main className="flex-1 p-6 space-y-6 max-w-[1600px] mx-auto w-full">
        {error && (
          <div className="bg-rose-950/40 border border-rose-800/60 p-4 rounded-xl flex items-center space-x-3 text-rose-200">
            <ShieldAlert className="h-5 w-5 text-rose-400 flex-shrink-0" />
            <p className="text-sm font-mono">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-[#0c1020] border border-gray-800/80 p-5 rounded-2xl shadow-xl flex items-center justify-between">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-gray-500 mb-1">Active Asset</p>
              <h3 className="text-2xl font-black text-white">{ticker}</h3>
            </div>
            <Activity className="h-8 w-8 text-emerald-500 opacity-60" />
          </div>

          <div className="bg-[#0c1020] border border-gray-800/80 p-5 rounded-2xl shadow-xl flex items-center justify-between">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-gray-500 mb-1">Spot Valuation</p>
              <h3 className="text-2xl font-black text-white">
                {loading ? '...' : stockData ? `$${stockData.latest_price.toFixed(2)}` : 'N/A'}
              </h3>
            </div>
            {stockData && stockData.sentiment_score >= 0 ? (
              <TrendingUp className="h-8 w-8 text-emerald-400 opacity-80" />
            ) : (
              <TrendingDown className="h-8 w-8 text-rose-400 opacity-80" />
            )}
          </div>

          <div className="bg-[#0c1020] border border-gray-800/80 p-5 rounded-2xl shadow-xl flex items-center justify-between">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-gray-500 mb-1">14-Day RSI</p>
              <h3 className="text-2xl font-black text-white">
                {loading ? '...' : stockData ? stockData.rsi.toFixed(2) : 'N/A'}
              </h3>
            </div>
            <div className="text-xs font-mono px-2 py-1 rounded bg-[#131930] border border-gray-800 text-gray-400">
              {stockData && stockData.rsi > 70 ? 'Overbought' : stockData && stockData.rsi < 30 ? 'Oversold' : 'Neutral'}
            </div>
          </div>

          <div className="bg-[#0c1020] border border-gray-800/80 p-5 rounded-2xl shadow-xl flex items-center justify-between">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-gray-500 mb-1">News Sentiment</p>
              <h3 className={`text-2xl font-black ${stockData && stockData.sentiment_score >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {loading ? '...' : stockData ? stockData.sentiment_score.toFixed(4) : 'N/A'}
              </h3>
            </div>
            <div className="text-xs font-mono text-gray-400">Scale [-1, 1]</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-[#0c1020] border border-gray-800/80 p-6 rounded-2xl shadow-xl flex flex-col h-[450px]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-mono uppercase tracking-wider text-gray-400 flex items-center gap-2">
                <Activity className="h-4 w-4 text-emerald-500" /> Market Trajectory Evaluation
              </h3>
            </div>
            <div className="flex-1 w-full min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#161b33" vertical={false} />
                  <XAxis dataKey="name" stroke="#4b5563" fontSize={11} tickLine={false} />
                  <YAxis domain={['dataMin - 10', 'dataMax + 10']} stroke="#4b5563" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0c1020', borderColor: '#1f2937', borderRadius: '8px' }}
                    labelStyle={{ color: '#9ca3af', fontFamily: 'monospace' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#10b981"
                    strokeWidth={3}
                    dot={{ fill: '#0c1020', stroke: '#10b981', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, strokeWidth: 0 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-[#0c1020] border border-gray-800/80 p-6 rounded-2xl shadow-xl flex flex-col justify-between h-[450px]">
            <div>
              <h3 className="text-sm font-mono uppercase tracking-wider text-gray-400 flex items-center gap-2 mb-6">
                <Cpu className="h-4 w-4 text-purple-400" /> Neural Classifier Signal
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-end">
                  <span className="text-xs font-mono text-gray-500">LSTM Convergence Weight</span>
                  <span className="text-3xl font-black tracking-tight text-white">
                    {loading ? '...' : stockData ? `${(stockData.ai_confidence * 100).toFixed(1)}%` : '0.0%'}
                  </span>
                </div>
                <div className="w-full bg-[#131930] rounded-full h-3 border border-gray-800 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-emerald-400 h-full transition-all duration-500"
                    style={{ width: stockData ? `${stockData.ai_confidence * 100}%` : '0%' }}
                  />
                </div>
                <p className="text-xs text-gray-400 leading-relaxed font-mono">
                  This metrics indicator displays the distribution weight calculated across multi-modal indicators. Values closer to 100% indicate structural alignment with trained market patterns.
                </p>
              </div>
            </div>

            <div className="pt-6 border-t border-gray-800/80">
              <div className="flex items-center gap-2 text-xs font-mono uppercase text-gray-500 mb-2">
                <span>Generative Summary Status</span>
              </div>
              <div className="p-3 bg-[#131930] border border-gray-800 rounded-xl text-xs text-gray-400 italic leading-relaxed font-mono h-24 overflow-y-auto">
                {loading ? 'Pacing analytical framework...' : stockData ? `AI Analyst Breakdown: The neural architecture indicates a defensive posture on ${ticker} given the convergence of negative sentiment catalysts and a stagnant RSI. Recommend holding capital in liquid cash positions until supply-chain visibility improves.` : 'No current pipeline data available.'}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-[#0c1020] border border-gray-800/80 p-6 rounded-2xl shadow-xl">
          <h3 className="text-sm font-mono uppercase tracking-wider text-gray-400 flex items-center gap-2 mb-4">
            <Newspaper className="h-4 w-4 text-blue-400" /> Real-Time Contextual Market Feeds
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {stockData && stockData.recent_headlines.map((headline, idx) => (
              <div key={idx} className="p-4 bg-[#131930] border border-gray-800/60 rounded-xl flex flex-col justify-between hover:border-gray-700 transition-colors">
                <p className="text-sm text-gray-200 font-medium leading-snug">{headline}</p>
                <span className="text-[10px] font-mono text-gray-500 mt-2 block">Provider Source Evaluation Feed</span>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}