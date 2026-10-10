import { redirect } from "next/navigation";

export function generateStaticParams() {
  return [{ "sign-up": [] }];
}

export default function SignUpPage() {
  redirect("/");
}
