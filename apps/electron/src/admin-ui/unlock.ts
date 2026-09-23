const create = new URLSearchParams(location.search).get('create') === '1';
const pw = document.getElementById('pw') as HTMLInputElement;
const pw2 = document.getElementById('pw2') as HTMLInputElement;
if (create) {
  document.getElementById('title')!.textContent = 'Set a vault password';
  pw2.style.display = '';
  pw2.required = true;
}
document.getElementById('f')!.addEventListener('submit', (e) => {
  e.preventDefault();
  if (create && pw.value !== pw2.value) {
    document.getElementById('err')!.textContent = 'Passwords do not match';
    return;
  }
  (window as any).faceid.submitPassword(pw.value);
});
