export function PlateText({ text }: { text: string }) {
  const parts = /^(\d{2})(الف|[\u0600-\u06ff])(\d{3})(\d{2})$/.exec(text);
  if (!parts) return <bdi dir="ltr">{text}</bdi>;
  return (
    <span dir="ltr" className="inline-flex flex-row gap-1" aria-label={text}>
      <bdi dir="ltr">{parts[1]}</bdi>
      <bdi dir="rtl">{parts[2]}</bdi>
      <bdi dir="ltr">{parts[3]}</bdi>
      <bdi dir="ltr">{parts[4]}</bdi>
    </span>
  );
}
