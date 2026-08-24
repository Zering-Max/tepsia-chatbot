"use client";

import { Button } from "./ui/button";
import { GitIcon } from "./icons";
import Link from "next/link";

export const Navbar = () => {
  return (
    <div className="sm:px-2 sm:pt-2 md:p-2 flex flex-row gap-2 justify-center">
      <Link href="https://github.com/vercel-labs/ai-sdk-preview-python-streaming">
        <Button variant="outline">
          <GitIcon /> Voir le projet open-source
        </Button>
      </Link>
    </div>
  );
};
