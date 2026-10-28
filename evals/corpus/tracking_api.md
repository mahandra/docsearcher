# Bluefin Tracking API v2 — Developer Guide

The Bluefin Tracking API lets partners create shipments, follow their status and receive notifications about delivery events. This guide describes version 2 of the API. Version 1 is deprecated and will be switched off on 31 January 2026.

## Base URL and Environments

All production requests are sent to https://api.bluefin-logistics.example/v2. A sandbox environment for testing is available at https://sandbox.bluefin-logistics.example/v2. Sandbox data is reset every Monday at 00:00 UTC, and shipments created in the sandbox are never delivered.

## Authentication

Every request must include an API key in the X-Api-Key header. API keys are created in the partner portal under Settings, then API Keys. Each partner account can have up to five active keys, which makes it possible to rotate a key without downtime: create a new key, deploy it, then revoke the old one.

API keys expire after 365 days. The portal sends a reminder e-mail to the account administrators thirty days before a key expires. A request with a missing, revoked or expired key is rejected with HTTP status 401 and error code AUTH_INVALID_KEY.

For server-to-server integrations that act on behalf of several partner accounts, OAuth 2.0 with the client credentials flow is also supported. Access tokens issued by the token endpoint are valid for one hour and must not be cached longer than that.

## Rate Limits

The API allows 600 requests per minute per API key. Bulk endpoints, whose paths end with /batch, count as ten requests each. Every response includes the headers X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset, which tell the client how many requests it may still send in the current window.

When the limit is exceeded, the API responds with HTTP status 429 and error code E-4291. The response contains a Retry-After header with the number of seconds to wait before retrying. Clients should back off exponentially and must not retry immediately in a loop; keys that keep exceeding the limit for more than ten minutes are temporarily blocked for one hour.

## Shipments

A shipment is created with POST /shipments. The request body contains the sender, the recipient, the parcels and the service level. Each parcel may weigh at most 31.5 kg, and a single shipment may contain at most 20 parcels. Larger consignments must be booked as freight through the Freight API.

The response contains a tracking number of the form BF followed by twelve digits, for example BF004512998731. The tracking number is the identifier used in all other endpoints.

To avoid creating duplicate shipments when a request is retried after a network error, clients should send an Idempotency-Key header with a unique value, such as a UUID. Requests with the same idempotency key received within 24 hours return the original response instead of creating a new shipment.

A shipment can be cancelled with DELETE /shipments/{tracking_number} as long as its status is CREATED. Once the parcels have been picked up, a cancellation is no longer possible and a return shipment must be created instead.

## Statuses

A shipment passes through the following statuses: CREATED, PICKED_UP, IN_TRANSIT, OUT_FOR_DELIVERY, DELIVERED. A shipment that cannot be delivered moves to the status EXCEPTION, with a reason code such as ADDRESS_UNKNOWN, RECIPIENT_ABSENT or DAMAGED. After three failed delivery attempts the shipment is returned to the sender and moves to the status RETURNED.

The current status is returned by GET /shipments/{tracking_number}. The full history of events, with timestamps in UTC and the location of each scan, is returned by GET /shipments/{tracking_number}/events.

## Pagination

List endpoints such as GET /shipments use cursor-based pagination. The limit parameter sets the page size: the default is 50 and the maximum is 200. Each response contains a next_cursor field; pass its value as the cursor parameter to fetch the next page. When next_cursor is null, the last page has been reached. Cursors expire after fifteen minutes.

## Webhooks

Instead of polling for status changes, partners can register a webhook URL in the partner portal. The API sends an HTTP POST request with a JSON body to this URL for every status change. The endpoint must be reachable over HTTPS and respond with a 2xx status code within ten seconds.

Each webhook request is signed. The X-Bluefin-Signature header contains an HMAC-SHA256 signature of the raw request body, computed with the webhook secret shown in the portal. Receivers must verify the signature and reject requests whose X-Bluefin-Timestamp header is older than five minutes, to prevent replay attacks.

If the endpoint does not respond with a 2xx status, the delivery is retried up to eight times with exponential backoff, starting after one minute and doubling each time. After the last failed attempt the event is dropped and the webhook is marked as failing in the portal. Webhooks that keep failing for three consecutive days are disabled automatically.

Events can be delivered more than once and not always in order. Receivers should use the event_id field to ignore duplicates and the occurred_at field to order events.

## Errors

Errors are returned as JSON objects with the fields code, message and request_id. The request_id should be included in every support ticket. Common errors are:

- E-4001: the request body is not valid JSON or a required field is missing.
- E-4002: the recipient address could not be validated.
- E-4041: the tracking number does not exist or belongs to another partner.
- E-4091: the shipment cannot be cancelled in its current status.
- E-4291: the rate limit was exceeded.
- E-5031: the service is temporarily unavailable; retry later.

## Data Retention

Shipment data, including addresses and the event history, is retained for 90 days after delivery. After that, personal data such as names, phone numbers and street addresses is anonymised, and only aggregated statistics are kept. Proof-of-delivery images, such as photos and signatures, are kept for 30 days after delivery.

## Versioning and Support

Backwards-incompatible changes are only introduced in a new major version of the API. Partners are informed about a new major version at least six months before the previous version is switched off. Technical questions can be sent to the integration support team through the partner portal; the team answers within one Business Day.
