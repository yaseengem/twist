
const initialState = {
  resume: null,
  jobDescription: null,
  score: null,
  suggestions: [],
};

export default function resumeReducer(state = initialState, action) {
  switch (action.type) {
    case 'SET_RESUME':
      return { ...state, resume: action.payload };
    case 'SET_JOB_DESCRIPTION':
      return { ...state, jobDescription: action.payload };
    case 'SET_SCORE':
      return { ...state, score: action.payload };
    case 'SET_SUGGESTIONS':
      return { ...state, suggestions: action.payload };
    default:
      return state;
  }
}