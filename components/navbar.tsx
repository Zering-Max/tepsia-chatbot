"use client";

import { Button } from "./ui/button";
// import { GitIcon } from "./icons";
import Link from "next/link";

export const Navbar = () => {
  const url = process.env.OPEN_SOURCE_URL
  return (
    <div className="p-2 flex flex-row gap-2 justify-center items-center">
      <span className="text-xs">
        Cet assistant conversationnel a été codé en
      </span>
      <Link href={`${url}`} target="_blank" className="text-xs underline underline-offset-4">
        open-source
      </Link>
      
    </div>
  );
};
