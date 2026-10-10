import { redirect } from "next/navigation";

export function generateStaticParams() {
  return [{ "sign-in": [] }];
}

export default function SignInPage() {
  redirect("/");
}
