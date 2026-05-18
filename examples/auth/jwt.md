# JWT Auth Example

Run the API with `AUTH_TYPE=jwt`, `AUTH_JWT_SECRET=<shared-secret>`, and
`AUTH_JWT_ALGORITHM=HS256`. Requests authenticate with
`Authorization: Bearer <jwt>` and the token `sub` claim becomes the user id.
