import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { AppLayout } from './components/AppLayout';
import { SingleCharPage } from './pages/SingleCharPage';
import { LineRecognitionPage } from './pages/LineRecognitionPage';
import { ParagraphRecognitionPage } from './pages/ParagraphRecognitionPage';
import { MathSolverPage } from './pages/MathSolverPage';
import { CharacterReferencePage } from './pages/CharacterReferencePage';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Navigate to="/v1" replace />} />
          <Route path="v1" element={<SingleCharPage />} />
          <Route path="v2" element={<LineRecognitionPage />} />
          <Route path="v3" element={<ParagraphRecognitionPage />} />
          <Route path="v4" element={<MathSolverPage />} />
          <Route path="character-reference" element={<CharacterReferencePage />} />
          <Route path="classes" element={<Navigate to="/character-reference" replace />} />
          <Route path="*" element={<Navigate to="/v1" replace />} />
        </Route>
      </Routes>
      <Toaster position="top-right" />
    </BrowserRouter>
  );
}

export default App;
