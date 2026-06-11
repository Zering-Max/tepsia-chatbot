import { motion } from "framer-motion";
import Link from "next/link";

// import { MessageIcon } from "./icons";
// import { LogoPython } from "@/app/icons";

export const Overview = () => {
  return (
    <motion.div
      key="overview"
      className="max-w-3xl mx-auto md:mt-20"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ delay: 0.5 }}
    >
      <div className="rounded-xl p-4 sm:p-6 flex flex-col gap-8 leading-relaxed text-center max-w-xl border border-white text-sm sm:text-base">
        {/* <p className="flex flex-row justify-center gap-4 items-center">
          <LogoPython size={32} />
          <span>+</span>
          <MessageIcon size={32} />
        </p> */}
        <p>
          Bonjour, je suis Teps’IA 👋 ! Un assistant conversationnel programmé pour vous aider à comprendre le projet de stockage de Gaz Naturel Liquéfié (GNL) porté par l’entreprise Tepsa au Nord de Strasbourg, et les raisons de s’y opposer. 
          Je suis mis à disposition gratuitement par le collectif citoyen Teps’out {" "}
          <Link
            className="font-medium underline underline-offset-4"
            href="https://www.tepsout.fr/"
            target="_blank"
          >
            (www.tepsout.fr)
          </Link>
        , mobilisé contre le projet.{" "}         
        </p>
        
      </div>
    </motion.div>
  );
};
