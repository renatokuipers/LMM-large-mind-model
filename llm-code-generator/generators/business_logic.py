from typing import Dict, List, Optional, Any, Tuple, Union, Set
import logging
import re
from pydantic import BaseModel, Field, validator

from schemas.code_entities import CodeComponent, ClassSignature, FunctionSignature
from schemas.llm_io import CodeGenerationResponse
from generators.base import BaseGenerator, GeneratorContext
from core.exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)


class BusinessLogicGeneratorContext(BaseModel):
    """Specialized context for business logic generation.
    
    Extends the standard GeneratorContext with business logic-specific
    information such as operations, permissions, and validation rules.
    """
    
    use_cases: List[str] = Field(
        default_factory=list, 
        description="Use cases this component should handle"
    )
    operations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Operations this component should implement"
    )
    permissions: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Permission requirements"
    )
    depends_on_repositories: List[str] = Field(
        default_factory=list, 
        description="Repository dependencies"
    )
    depends_on_services: List[str] = Field(
        default_factory=list, 
        description="Service dependencies"
    )
    validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Business validation rules"
    )
    
    def to_generator_context(self) -> GeneratorContext:
        """Convert to a standard generator context.
        
        Returns:
            Standard generator context
        """
        # Process operations to add to requirements
        operation_requirements = []
        for op in self.operations:
            op_name = op.get("name", "unknown")
            op_desc = op.get("description", "")
            op_params = op.get("parameters", [])
            op_result = op.get("result", "Any")
            
            params_str = ""
            if op_params:
                params_list = []
                for param in op_params:
                    param_name = param.get("name", "unknown")
                    param_type = param.get("type", "Any")
                    params_list.append(f"{param_name}: {param_type}")
                params_str = f" taking ({', '.join(params_list)})"
            
            op_req = f"Implement operation '{op_name}'{params_str} returning {op_result}{f': {op_desc}' if op_desc else ''}"
            operation_requirements.append(op_req)
        
        # Process use cases
        use_case_requirements = []
        for use_case in self.use_cases:
            use_case_requirements.append(f"Support use case: {use_case}")
        
        # Process validation rules
        validation_requirements = []
        for rule in self.validation_rules:
            rule_desc = rule.get("description", "unknown")
            validation_requirements.append(f"Enforce business rule: {rule_desc}")
        
        # Process dependencies
        dependencies = []
        for repo in self.depends_on_repositories:
            dependencies.append(repo)
        for service in self.depends_on_services:
            dependencies.append(service)
        
        # Combine all requirements
        all_requirements = []
        if use_case_requirements:
            all_requirements.extend(use_case_requirements)
        
        if operation_requirements:
            all_requirements.append("Operations:")
            all_requirements.extend([f"- {req}" for req in operation_requirements])
        
        if validation_requirements:
            all_requirements.append("Business Rules:")
            all_requirements.extend([f"- {req}" for req in validation_requirements])
        
        if self.permissions:
            all_requirements.append("Permission Requirements:")
            for perm in self.permissions:
                perm_desc = perm.get("description", "unknown")
                all_requirements.append(f"- {perm_desc}")
        
        # Create standard generator context
        return GeneratorContext(
            component_type="class",
            name=self.name,
            module_path=self.module_path,
            description=self.description,
            requirements=all_requirements,
            dependencies=dependencies,
            additional_context=self.additional_context,
            project_description=self.project_description
        )


class BusinessLogicGenerator(BaseGenerator):
    """Generator for business logic components.
    
    This generator specializes in creating service classes, use cases,
    and business rule implementations with proper dependency injection.
    """
    
    async def generate(self, context: GeneratorContext) -> CodeComponent:
        """Generate a business logic component.
        
        Args:
            context: Generation context
            
        Returns:
            Generated component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating business logic component: {context.module_path}.{context.name}")
        
        # Add business logic specific guidance to context
        if context.additional_context:
            context.additional_context += self._get_business_logic_guidance()
        else:
            context.additional_context = self._get_business_logic_guidance()
        
        # Generate code
        try:
            response = await self._generate_with_llm(context)
            
            # Enhance generated code
            enhanced_code = self._enhance_business_logic(response.code)
            response.code = enhanced_code
            
            # Create component
            component = self._create_component_from_response(context, response)
            
            # Validate component
            self._validate_component(component)
            
            return component
            
        except Exception as e:
            logger.error(f"Error generating business logic component {context.name}: {str(e)}")
            raise
    
    def _get_business_logic_guidance(self) -> str:
        """Get guidance for business logic generation.
        
        Returns:
            Guidance text
        """
        return """
        ## Business Logic Guidelines
        
        When implementing business logic, follow these guidelines:
        
        1. Use dependency injection for all dependencies
        2. Implement clear separation of concerns
        3. Use Pydantic models for input/output validation
        4. Implement proper error handling with custom exceptions
        5. Add transaction management where appropriate
        6. Include comprehensive logging
        7. Implement all required business rules as validators
        8. Use type hints consistently
        9. Add proper docstrings for all methods
        10. Prefer composition over inheritance
        11. Implement proper permission checks
        12. Return explicit result types
        
        Example Service Pattern:
        
        ```python
        class UserService:
            """Service for managing user operations.
            
            This service handles business logic related to user management,
            including registration, profile updates, and authentication.
            """
            
            def __init__(
                self, 
                user_repository: UserRepository,
                email_service: EmailService,
                auth_service: AuthService
            ):
                """Initialize the service with dependencies.
                
                Args:
                    user_repository: Repository for user data access
                    email_service: Service for sending emails
                    auth_service: Service for authentication
                """
                self.user_repository = user_repository
                self.email_service = email_service
                self.auth_service = auth_service
                self.logger = logging.getLogger(__name__)
            
            async def register_user(self, user_data: UserCreate) -> UserResponse:
                """Register a new user.
                
                Args:
                    user_data: User registration data
                    
                Returns:
                    Created user
                    
                Raises:
                    UserAlreadyExistsError: If a user with the same email exists
                    ValidationError: If the data is invalid
                """
                # Validate business rules
                self._validate_password_strength(user_data.password)
                
                # Check for existing user
                existing_user = await self.user_repository.get_by_email(user_data.email)
                if existing_user:
                    raise UserAlreadyExistsError(f"User with email {user_data.email} already exists")
                
                # Create user
                hashed_password = self.auth_service.hash_password(user_data.password)
                user = User(
                    email=user_data.email,
                    hashed_password=hashed_password,
                    full_name=user_data.full_name
                )
                
                # Persist user
                created_user = await self.user_repository.create(user)
                
                # Send welcome email
                await self.email_service.send_welcome_email(user.email)
                
                # Log action
                self.logger.info(f"User registered: {user.email}")
                
                # Return user response
                return UserResponse.from_orm(created_user)
            
            def _validate_password_strength(self, password: str) -> None:
                """Validate password strength according to business rules.
                
                Args:
                    password: Password to validate
                    
                Raises:
                    ValidationError: If the password doesn't meet requirements
                """
                if len(password) < 8:
                    raise ValidationError("Password must be at least 8 characters long")
                
                if not any(c.isupper() for c in password):
                    raise ValidationError("Password must contain at least one uppercase letter")
                
                if not any(c.isdigit() for c in password):
                    raise ValidationError("Password must contain at least one digit")
        ```
        """
    
    def _enhance_business_logic(self, code: str) -> str:
        """Enhance business logic in the generated code.
        
        This method post-processes the generated code to ensure
        best practices are followed in business logic implementation.
        
        Args:
            code: Generated code
            
        Returns:
            Enhanced code
        """
        # Ensure proper exception handling
        if 'raise' in code and 'except' in code:
            # Look for generic exceptions
            if re.search(r'except Exception', code):
                # Add more specific exception handling
                code = re.sub(
                    r'except Exception as e:',
                    'except ValidationError as e:\n        # Handle validation errors\n        self.logger.error(f"Validation error: {str(e)}")\n        raise\n        \n    except Exception as e:',
                    code
                )
            
            # Look for missing logging in exception handlers
            exception_blocks = re.finditer(r'except ([A-Za-z0-9_]+) as e:(.*?)(?=\n\s*?(?:except|finally|\S)|\Z)', code, re.DOTALL)
            for match in exception_blocks:
                exception_block = match.group(2)
                if 'logger' not in exception_block and 'log' not in exception_block:
                    # Add logging
                    indent = re.match(r'(\s+)', exception_block).group(1) if re.match(r'(\s+)', exception_block) else '        '
                    exception_type = match.group(1)
                    log_line = f"{indent}self.logger.error(f\"{exception_type}: {{str(e)}}\")\n"
                    code = code.replace(exception_block, log_line + exception_block)
        
        # Ensure logging is set up
        if 'logger' in code and 'logging.getLogger' not in code:
            # Add import if needed
            if 'import logging' not in code:
                code = 'import logging\n' + code
            
            # Add logger initialization to __init__ method
            init_pattern = re.compile(r'def __init__\(self(.*?)\):(.*?)(?=\n\s*?def|\Z)', re.DOTALL)
            init_match = init_pattern.search(code)
            
            if init_match:
                init_body = init_match.group(2)
                if 'self.logger' not in init_body:
                    # Add logger initialization to __init__
                    indent = re.match(r'(\s+)', init_body).group(1) if re.match(r'(\s+)', init_body) else '        '
                    logger_line = f"\n{indent}self.logger = logging.getLogger(__name__)"
                    code = code.replace(init_body, init_body + logger_line)
            
            # If no __init__ method, add one
            if 'def __init__' not in code:
                class_pattern = re.compile(r'class\s+(\w+).*?:', re.DOTALL)
                class_match = class_pattern.search(code)
                
                if class_match:
                    class_name = class_match.group(1)
                    class_pos = class_match.end()
                    
                    # Find first method to determine indentation
                    method_pattern = re.compile(r'\n(\s+)def\s+\w+\(', re.DOTALL)
                    method_match = method_pattern.search(code[class_pos:])
                    
                    if method_match:
                        indent = method_match.group(1)
                        init_method = f"\n{indent}def __init__(self):\n{indent}    \"\"\"Initialize the {class_name}.\"\"\"\n{indent}    self.logger = logging.getLogger(__name__)"
                        code = code[:class_pos] + init_method + code[class_pos:]
        
        # Check for missing type hints
        method_pattern = re.compile(r'def\s+(\w+)\s*\((.*?)\)(?:\s*->.*?)?:', re.DOTALL)
        for match in method_pattern.finditer(code):
            method_name = match.group(1)
            params = match.group(2)
            
            # Skip dunder methods
            if method_name.startswith('__') and method_name.endswith('__'):
                continue
            
            # Check if method already has return type hint
            method_sig = match.group(0)
            if '->' not in method_sig and method_name != '__init__':
                # Try to infer appropriate return type
                if method_name.startswith('get_') or method_name.startswith('find_'):
                    return_type = ' -> Optional[Any]'
                elif method_name.startswith('list_') or method_name.startswith('search_'):
                    return_type = ' -> List[Any]'
                elif method_name.startswith('create_'):
                    return_type = ' -> Any'
                elif method_name.startswith('update_'):
                    return_type = ' -> Optional[Any]'
                elif method_name.startswith('delete_') or method_name.startswith('remove_'):
                    return_type = ' -> bool'
                elif method_name.startswith('check_') or method_name.startswith('validate_') or method_name.startswith('is_'):
                    return_type = ' -> bool'
                elif method_name.startswith('process_') or method_name.startswith('handle_'):
                    return_type = ' -> Any'
                elif method_name.startswith('_'):  # Private helper method
                    return_type = ' -> None'
                else:
                    return_type = ' -> Any'
                
                # Add return type hint
                new_sig = f"def {method_name}({params}){return_type}:"
                code = code.replace(method_sig, new_sig)
            
            # Check if parameters have type hints
            param_list = [p.strip() for p in params.split(',') if p.strip()]
            for i, param in enumerate(param_list):
                if param == 'self':
                    continue
                
                if ':' not in param:
                    # Add type hint
                    param_name = param.strip()
                    param_with_hint = f"{param_name}: Any"
                    param_list[i] = param_with_hint
            
            # Replace parameters with type-hinted ones
            if param_list and any(':' not in p for p in param_list if p != 'self'):
                new_params = ', '.join(param_list)
                new_sig = method_sig.replace(params, new_params)
                code = code.replace(method_sig, new_sig)
        
        # Ensure Any is imported if we added it
        if 'Any' in code and 'from typing import' in code:
            if 'Any' not in code.split('from typing import')[1].split('\n')[0]:
                code = code.replace('from typing import', 'from typing import Any, ')
        elif 'Any' in code:
            code = 'from typing import Any, Optional, List\n' + code
        
        # Ensure Optional is imported if used
        if 'Optional' in code and 'from typing import' in code:
            if 'Optional' not in code.split('from typing import')[1].split('\n')[0]:
                code = code.replace('from typing import', 'from typing import Optional, ')
        elif 'Optional' in code and 'from typing import' not in code:
            code = 'from typing import Optional, Any, List\n' + code
        
        # Ensure List is imported if used
        if 'List' in code and 'from typing import' in code:
            if 'List' not in code.split('from typing import')[1].split('\n')[0]:
                code = code.replace('from typing import', 'from typing import List, ')
        elif 'List' in code and 'from typing import' not in code:
            code = 'from typing import List, Any, Optional\n' + code
        
        return code
    
    async def generate_service(self, 
                            context: GeneratorContext, 
                            repository_signatures: List[ClassSignature]) -> CodeComponent:
        """Generate a service component that works with repositories.
        
        Args:
            context: Generation context
            repository_signatures: Signatures of repositories this service will use
            
        Returns:
            Generated service component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating service: {context.module_path}.{context.name}")
        
        # Add repository info to context
        repo_info = "## Repository Dependencies\n\nThis service should use the following repositories:\n\n"
        
        for repo in repository_signatures:
            repo_info += f"### {repo.name}\n\n"
            repo_info += "Methods:\n"
            
            for method in repo.methods:
                # Format method signature
                params = [str(param) for param in method.parameters[1:]]  # Skip self
                params_str = ", ".join(params)
                return_str = f" -> {method.return_type.type_hint}" if method.return_type else ""
                
                method_sig = f"- `{method.name}({params_str}){return_str}`"
                if method.docstring:
                    method_sig += f": {method.docstring.summary}"
                
                repo_info += method_sig + "\n"
        
        # Add to context
        if context.additional_context:
            context.additional_context += "\n\n" + repo_info
        else:
            context.additional_context = repo_info
        
        # Generate service
        return await self.generate(context)
    
    async def generate_use_case(self, 
                             context: GeneratorContext,
                             service_dependencies: List[ClassSignature]) -> CodeComponent:
        """Generate a use case component that orchestrates services.
        
        Args:
            context: Generation context
            service_dependencies: Signatures of services this use case will orchestrate
            
        Returns:
            Generated use case component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating use case: {context.module_path}.{context.name}")
        
        # Add use case specific guidance
        use_case_guidance = """
        ## Use Case Pattern Guidelines
        
        When implementing use cases, follow these guidelines:
        
        1. Each use case should focus on a single business operation
        2. Use cases should orchestrate services and don't have business logic
        3. Use cases should not access repositories directly
        4. Use Pydantic models for input/output
        5. Each use case should have a clear, descriptive name
        6. Use cases should handle all exceptions from services they use
        7. Follow the Command pattern for write operations
        8. Follow the Query pattern for read operations
        9. Consider implementing as functions rather than classes for simplicity
        10. Add proper error reporting
        
        Example Use Case:
        
        ```python
        class CreateOrderUseCase:
            """Use case for creating a new order in the system.
            
            This use case orchestrates the process of creating an order,
            including inventory validation, payment processing, and notification.
            """
            
            def __init__(
                self,
                order_service: OrderService,
                inventory_service: InventoryService,
                payment_service: PaymentService,
                notification_service: NotificationService
            ):
                """Initialize the use case with required services.
                
                Args:
                    order_service: Service for order management
                    inventory_service: Service for inventory management
                    payment_service: Service for payment processing
                    notification_service: Service for sending notifications
                """
                self.order_service = order_service
                self.inventory_service = inventory_service
                self.payment_service = payment_service
                self.notification_service = notification_service
                self.logger = logging.getLogger(__name__)
            
            async def execute(self, command: CreateOrderCommand) -> OrderResponse:
                """Execute the use case to create a new order.
                
                Args:
                    command: Order creation command with all required data
                    
                Returns:
                    Created order details
                    
                Raises:
                    OutOfStockError: If products are not available
                    PaymentError: If payment processing fails
                    ValidationError: If the order data is invalid
                """
                try:
                    # Check inventory availability
                    for item in command.items:
                        available = await self.inventory_service.check_availability(
                            item.product_id, item.quantity
                        )
                        if not available:
                            raise OutOfStockError(f"Product {item.product_id} is out of stock")
                    
                    # Create order
                    order = await self.order_service.create_order(
                        user_id=command.user_id,
                        items=command.items,
                        shipping_address=command.shipping_address
                    )
                    
                    # Process payment
                    payment = await self.payment_service.process_payment(
                        order_id=order.id,
                        amount=order.total_amount,
                        payment_method=command.payment_method
                    )
                    
                    # Update order with payment info
                    order = await self.order_service.update_payment_status(
                        order_id=order.id,
                        payment_id=payment.id,
                        status=payment.status
                    )
                    
                    # Reserve inventory
                    for item in order.items:
                        await self.inventory_service.reserve_inventory(
                            product_id=item.product_id,
                            quantity=item.quantity,
                            order_id=order.id
                        )
                    
                    # Send confirmation notification
                    await self.notification_service.send_order_confirmation(
                        user_id=command.user_id,
                        order_id=order.id
                    )
                    
                    # Return response
                    return OrderResponse.from_orm(order)
                    
                except OutOfStockError as e:
                    self.logger.error(f"Out of stock error: {str(e)}")
                    raise
                    
                except PaymentError as e:
                    self.logger.error(f"Payment error: {str(e)}")
                    raise
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error creating order: {str(e)}")
                    raise OrderCreationError(f"Failed to create order: {str(e)}")
        ```
        """
        
        # Add service dependencies to context
        service_info = "## Service Dependencies\n\nThis use case should use the following services:\n\n"
        
        for service in service_dependencies:
            service_info += f"### {service.name}\n\n"
            service_info += "Methods:\n"
            
            for method in service.methods:
                # Format method signature
                params = [str(param) for param in method.parameters[1:]]  # Skip self
                params_str = ", ".join(params)
                return_str = f" -> {method.return_type.type_hint}" if method.return_type else ""
                
                method_sig = f"- `{method.name}({params_str}){return_str}`"
                if method.docstring:
                    method_sig += f": {method.docstring.summary}"
                
                service_info += method_sig + "\n"
        
        # Add to context
        if context.additional_context:
            context.additional_context += "\n\n" + use_case_guidance + "\n\n" + service_info
        else:
            context.additional_context = use_case_guidance + "\n\n" + service_info
        
        # Generate use case
        return await self.generate(context)