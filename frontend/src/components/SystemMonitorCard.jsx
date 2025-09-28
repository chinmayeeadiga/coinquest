

import React from "react";

function SystemMonitorCard() {
    const system = {
    cpu: 27000,
    memory: -2.5,
    network: 3500,
    uptime: 500000,
  };

  return (
    <div className="absolute bottom-4 right-4 border border-[#ff6f61] rounded text-white  p-4 w-60 shadow-lg">
      <h2 className="text-lg font-bold">SYSTEM MONITOR</h2>
      <p>CPU Usage: {system.cpu}%</p>
      <p>Memory: {system.memory} GB</p>
      <p>Network: {system.network} MB/s</p>
      <p>Uptime: {system.uptime}</p>
      <pre className="bg-white/30 text-xs mt-2 p-2 rounded">
        {system.logs}
      </pre>
    </div>
  );
}

export default SystemMonitorCard;
