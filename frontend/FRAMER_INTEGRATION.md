# Framer Integration Guide

This guide will help you integrate your Framer template into this React project.

## Quick Start with Your Framer Template

Your Framer project: https://creative-assumptions-648427.framer.app/

### Step 1: Export Components from Framer

1. **Open your Framer project** at https://creative-assumptions-648427.framer.app/
2. **Install the React Export plugin**:
   - Go to the Framer plugin store
   - Search for "React Export"
   - Install the plugin

3. **Select components to export**:
   - Choose the components you want to use in React
   - Recommended: Header, Footer, Cards, Forms, Buttons

4. **Export the components**:
   - Use the React Export plugin
   - Download the generated .zip file
   - Extract to a temporary folder

### Step 2: Import into This Project

1. **Copy component files**:
   ```bash
   # Copy your exported components to src/components/framer/
   cp -r path/to/exported/components/* src/components/framer/
   ```

2. **Install any additional dependencies** (if required by your components):
   ```bash
   npm install [any-additional-deps]
   ```

### Step 3: Enhance with API Integration

Use the provided helper functions to enhance your Framer components:

```tsx
// Example: Enhance a Framer form component
import { withApiIntegration } from '@/components/FramerHelpers';
import { YourFramerForm } from '@/components/framer/YourFramerForm';

const EnhancedForm = withApiIntegration(YourFramerForm);

// Use in your app
function ContactPage() {
  const handleSubmit = async (formData) => {
    // This will automatically call your API
    console.log('Form submitted:', formData);
  };

  return (
    <EnhancedForm
      onSubmit={handleSubmit}
      isLoading={false}
    />
  );
}
```

### Step 4: Connect to Backend APIs

Your components can now easily connect to backend services:

```tsx
import { useFormSubmission } from '@/hooks/useApi';

function YourComponent() {
  const { submitForm, isSubmitting, submitError } = useFormSubmission();

  const handleFormSubmit = async (data) => {
    const success = await submitForm('/contact', data);
    if (success) {
      // Handle success
    }
  };

  return (
    <YourFramerComponent
      onSubmit={handleFormSubmit}
      isLoading={isSubmitting}
      error={submitError}
    />
  );
}
```

## Common Integration Patterns

### 1. Data-Driven Components

```tsx
import { useApi } from '@/hooks/useApi';

function DataDrivenCard() {
  const { data, loading, error } = useApi(() => 
    fetch('/api/data').then(res => res.json())
  );

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;

  return (
    <YourFramerCard
      title={data.title}
      description={data.description}
      imageUrl={data.imageUrl}
    />
  );
}
```

### 2. Form Handling

```tsx
function ContactForm() {
  const { submitForm, isSubmitting } = useFormSubmission();

  return (
    <YourFramerContactForm
      onSubmit={(data) => submitForm('/contact', data)}
      isLoading={isSubmitting}
    />
  );
}
```

### 3. User Authentication

```tsx
import { useAuth } from '@/hooks/useApi';

function UserProfile() {
  const { user, isAuthenticated, logout } = useAuth();

  if (!isAuthenticated) {
    return <YourFramerLoginForm />;
  }

  return (
    <YourFramerProfileCard
      user={user}
      onLogout={logout}
    />
  );
}
```

## Troubleshooting

### Common Issues

1. **Component not rendering**:
   - Check import paths
   - Ensure all dependencies are installed
   - Verify component exports

2. **Styling issues**:
   - Make sure Tailwind CSS is working
   - Check for conflicting styles
   - Use browser dev tools to debug

3. **TypeScript errors**:
   - Add proper type definitions
   - Use the provided `FramerComponentProps` interface
   - Check `src/types/index.ts` for available types

### Getting Help

1. Check the main README.md for detailed documentation
2. Look at example components in `src/components/framer/ExampleComponents.tsx`
3. Review the API integration patterns in `src/hooks/useApi.ts`

## Next Steps

1. **Export your Framer components** using the steps above
2. **Replace the example components** with your actual Framer exports
3. **Customize the API endpoints** in `src/api/service.ts`
4. **Style your components** using Tailwind CSS classes
5. **Test your integration** with the development server

```bash
npm run dev
```

Your application will be available at http://localhost:3000

Happy coding! 🚀