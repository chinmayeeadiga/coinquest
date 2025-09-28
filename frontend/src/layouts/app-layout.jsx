import React from "react";
import { Outlet } from "react-router-dom";
import CyberpunkBackground from "../components/ui/cyberpunk";
import AuroraBackground from "../components/ui/aurora-background";

const AppLayout = () => {
  return (
    <>
   <main className="relative min-h-screen">
      <CyberpunkBackground className="fixed top-0 left-0 w-screen h-screen -z-10" />
      
      <div className="relative z-10">
        <Outlet />
      </div>
    </main>
    </>
  );
};

export default AppLayout;
