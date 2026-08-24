"use client";

import { Button } from "./ui/button";
import { GitIcon } from "./icons";
import Link from "next/link";

export const Navbar = () => {
  const url = process.env.OPEN_SOURCE_URL
  return (
    <div className="p-2 flex flex-row gap-2 justify-center">
      <Link href={`${url}`} target="_blank">
        <Button variant="outline">
          <GitIcon /> Voir le projet open-source
        </Button>
      </Link>
    </div>
  );
};
