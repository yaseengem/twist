
import { combineReducers } from 'redux';
import resumeReducer from './resumeReducer';

export default combineReducers({
  resume: resumeReducer,
});