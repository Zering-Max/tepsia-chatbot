import { motion } from "framer-motion";
import Link from "next/link";


export const SourcesDisclaimer = () => {
  return (
    <motion.div
      key="sources-disclaimer"
      className="max-w-3xl mx-auto mb-4"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ delay: 0.5 }}
    >
      <div className="rounded-xl p-2 sm:p-4 flex flex-col gap-8 leading-relaxed text-center max-w-xl border border-white text-xs sm:text-sm">
       
        <p>
          Tous les documents qui servent de sources à cet assistant sont accessibles sur notre site {" "}
          <Link
            className="font-medium underline underline-offset-4"
            href="https://www.tepsout.fr/"
            target="_blank"
          >
            www.tepsout.fr
          </Link>
        - l&apos;IA est un outil puissant mais aux conséquences écologiques et sociales très préoccupantes, à utiliser avec modération.{" "}         
        </p>
        
      </div>
    </motion.div>
  );
};
