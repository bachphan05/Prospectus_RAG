import { getValue, getFieldInfo, isEffectivelyNA } from './utils';

/**
 * DataField Component - Display/Edit field with hover functionality
 * Moved outside Dashboard to prevent losing focus on input
 */
const DataField = ({ label, field, fieldName, editable, isEditMode, editedData, hoveredField, setHoveredField, updateEditedField }) => {
  // Get the current value based on edit mode
  const getCurrentValue = () => {
    if (!isEditMode) return getValue(field);
    
    // Navigate through editedData using fieldName path
    const keys = fieldName.split('.');
    let current = editedData;
    for (const key of keys) {
      if (!current) return '';
      current = current[key];
    }
    return getValue(current);
  };
  
  const displayValue = getCurrentValue();
  const info = getFieldInfo(field);
  
  const handleMouseEnter = () => {
    if (info && !isEditMode) {
      setHoveredField({ fieldName, page: info.page, bbox: info.bbox });
    }
  };
  
  const handleMouseLeave = () => {
    setHoveredField(null);
  };
  
  const isHovered = hoveredField?.fieldName === fieldName;

  // In view mode: hide fields that are effectively N/A.
  // In edit mode: always show so users can fill missing values.
  if (!isEditMode && isEffectivelyNA(displayValue)) {
    return null;
  }
  
  if (isEditMode && editable) {
    return (
      <div>
        <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</dt>
        <dd className="mt-1">
          <input
            type="text"
            value={displayValue || ''}
            onChange={(e) => updateEditedField(fieldName, e.target.value)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </dd>
      </div>
    );
  }
  
  return (
    <div>
      <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</dt>
      <dd 
        className={`mt-1 text-sm text-gray-900 transition-all group ${
          info ? 'cursor-pointer hover:bg-yellow-100 hover:shadow-sm px-2 py-1 -mx-2 -my-1 rounded' : ''
        } ${isHovered ? 'bg-yellow-200 shadow-md' : ''}`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <span>{displayValue}</span>
        {info && (
          <span className="ml-2 text-xs text-gray-500 italic font-normal opacity-0 group-hover:opacity-100 transition-opacity">
            (Page {info.page})
          </span>
        )}
      </dd>
    </div>
  );
};

export default DataField;
